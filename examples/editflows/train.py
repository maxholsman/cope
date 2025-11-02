# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os

import torch
import torch.distributed as dist
from data import data
from flow_matching.loss import MixturePathGeneralizedKL, EditFlowsLoss

from logic import evaluate, flow, generate, training
from logic.state import TrainState
from model import Transformer
from omegaconf import OmegaConf
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2TokenizerFast, EsmTokenizer, AutoTokenizer, AutoModelForCausalLM, EsmForMaskedLM
from utils import checkpointing, logging


def run_train(rank: int, cfg: OmegaConf) -> None:
    torch.manual_seed(cfg.training.seed + rank)

    # Logging and configuration
    work_dirs = checkpointing.get_work_dirs(work_dir=cfg.work_dir, rank=rank)

    logger = logging.TrainLogger(log_dir=work_dirs.root, rank=rank, cfg=cfg)
    logger.info(work_dirs)
    logger.info(cfg)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.log_devices(device=device, logger=logger)

    # Data
    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer_name = getattr(cfg.data, "tokenizer", "facebook/esm2_t12_35M_UR50D")
    tokenizer = EsmTokenizer.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    # Get all special token IDs from tokenizer
    special_token_ids = []
    
    # Try to extract from special_tokens_map (works well for ESM tokenizers)
    if hasattr(tokenizer, "special_tokens_map") and tokenizer.special_tokens_map:
        for token_name, token_value in tokenizer.special_tokens_map.items():
            if isinstance(token_value, str):
                token_id = tokenizer.convert_tokens_to_ids(token_value)
                if isinstance(token_id, list) and len(token_id) > 0:
                    token_id = token_id[0]
                elif not isinstance(token_id, (int, type(None))):
                    continue
                if token_id is not None and token_id not in special_token_ids:
                    special_token_ids.append(token_id)
    
    # Fallback to manual extraction from individual token_id properties
    if len(special_token_ids) == 0:
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            special_token_ids.append(tokenizer.pad_token_id)
        if hasattr(tokenizer, "cls_token_id") and tokenizer.cls_token_id is not None:
            special_token_ids.append(tokenizer.cls_token_id)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            special_token_ids.append(tokenizer.eos_token_id)
        if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
            special_token_ids.append(tokenizer.unk_token_id)
        if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
            special_token_ids.append(tokenizer.mask_token_id)
        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            special_token_ids.append(tokenizer.bos_token_id)
        if hasattr(tokenizer, "sep_token_id") and tokenizer.sep_token_id is not None:
            special_token_ids.append(tokenizer.sep_token_id)
    
    # Additional tokens to exclude: punctuation and null tokens (e.g., '.', '-', '<null_1>')
    # These are typically ID 29, 30, 31 for ESM tokenizers but we extract them dynamically
    additional_excluded_tokens = ['.', '-', '<null_1>']
    for token_str in additional_excluded_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if isinstance(token_id, list) and len(token_id) > 0:
                token_id = token_id[0]
            if isinstance(token_id, int) and token_id not in special_token_ids:
                special_token_ids.append(token_id)
        except (ValueError, KeyError):
            # Token not found in vocabulary, skip
            pass

    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=special_token_ids
    )

    # Model initialization
    model = Transformer(
        config=cfg.model, vocab_size=vocab_size, masked=source_distribution.masked
    ).to(device)

    num_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in the model: {num_parameters}")

    model = DDP(model, device_ids=[rank], static_graph=False, find_unused_parameters=True)  # find_unused_parameters=True needed when ESM is frozen
    logger.info(model)

    # Optimizer initialization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
        fused=cfg.optim.fused,
    )
    logger.info(f"Optimizer: {optimizer}")
    scaler = torch.amp.GradScaler("cuda")
    logger.info(f"Scaler: {scaler}")

    data_state = data.get_data_state(config=cfg)

    # Train state
    state = TrainState(model=model, optimizer=optimizer, step=1, data_state=data_state)
    state.restore_checkpoint(ckpt_dir=work_dirs.checkpoint, device=device, rank=rank)

    train_iter, eval_iter = data.get_data_loaders(config=cfg, data_state=data_state)

    if cfg.model.compile:
        state.compile_model()
        torch.set_float32_matmul_precision("high")

    # Flow matching
    eps_id = getattr(cfg.flow, "eps_id", -1)
    path = flow.get_path(
        scheduler_type=cfg.flow.scheduler_type, exponent=cfg.flow.exponent, eps_id=eps_id
    )
    loss_fn = flow.get_loss_function(loss_function=cfg.flow.loss_function, path=path)
    # Elbo may have singularity at 1
    time_epsilon = 1e-3 if isinstance(loss_fn, EditFlowsLoss) else 0.0

    num_train_steps = cfg.optim.n_iters
    logger.info(f"Starting training loop at step {state.step}.")

    train_loss_values = []

    while state.step <= num_train_steps:
        loss = training.step(
            loss_fn=loss_fn,
            path=path,
            state=state,
            scaler=scaler,
            iterator=train_iter,
            optim_params=cfg.optim,
            device=device,
            source_distribution=source_distribution,
            logger=logger,
            training=True,
            time_epsilon=time_epsilon,
        )

        train_loss_values.append(loss)

        # Train logging
        if state.step % cfg.logging.log_freq == 0:
            agg_train_loss_values = torch.tensor(
                train_loss_values, device=device
            ).mean()
            dist.all_reduce(agg_train_loss_values, dist.ReduceOp.AVG)
            logger.log_metric(
                value=agg_train_loss_values, name="Loss", stage="Train", step=state.step
            )

            train_loss_values = []

        # Checkpoint
        if state.step % cfg.training.snapshot == 0:
            logger.info("Saving checkpoint...", step=state.step)

            state.save_checkpoint(ckpt_dir=work_dirs.checkpoint, rank=rank)

        # Evaluation loss
        if state.step % cfg.training.eval_freq == 0:
            logger.info("Evaluating loss...", step=state.step)

            eval_loss = training.step(
                state=state,
                loss_fn=loss_fn,
                path=path,
                scaler=scaler,
                iterator=eval_iter,
                device=device,
                source_distribution=source_distribution,
                logger=logger,
                training=False,
                time_epsilon=time_epsilon,
            )

            dist.all_reduce(eval_loss, dist.ReduceOp.AVG)
            logger.log_metric(
                value=eval_loss.item(), name="Loss", stage="Evaluation", step=state.step
            )

        # Generation
        if state.step % cfg.training.perplexity_freq == 0 and getattr(cfg.eval, "perplexity", True):
            state.eval()

            logger.info("Generating text...", step=state.step)

            samples = generate.generate_samples(
                model=state.model,
                step=state.step,
                sample_dir=work_dirs.samples,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                rank=rank,
                device=device,
                path=path,
                source_distribution=source_distribution,
                sample_batch_size=cfg.eval.sample_batch_size,
                sequence_length=cfg.model.length,
                sampling_steps=cfg.flow.sampling_steps,
                time_epsilon=time_epsilon,
            )

            # Use ESM model for perplexity (samples already use ESM tokenizer)
            lm_name = getattr(cfg.eval, "perplexity_lm", tokenizer_name)  # Use same tokenizer by default
            lm_model = EsmForMaskedLM.from_pretrained(lm_name).to(device).eval()
            
            # Samples are already tokenized with ESM tokenizer, so use them directly
            # Convert samples to list if needed
            if isinstance(samples, list):
                lm_samples = [s.to(device) for s in samples]
            else:
                lm_samples = [samples[i].to(device) for i in range(samples.size(0))]
            
            lm_pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0

            perplexity = evaluate.compute_perplexity(
                samples=lm_samples,           # ragged list for the LM
                lm_model=lm_model,
                pad_id=lm_pad_id,
                batch_size=cfg.eval.perplexity_batch_size,
            )
            dist.all_reduce(perplexity, dist.ReduceOp.AVG)
            logger.log_metric(
                value=perplexity, name="Perplexity", stage="Evaluation", step=state.step
            )

            entropy = evaluate.compute_entropy(samples=samples)
            dist.all_reduce(entropy, dist.ReduceOp.AVG)

            logger.log_metric(
                value=entropy, name="Entropy", stage="Evaluation", step=state.step
            )

            dist.barrier()

        state.step = state.step + 1

    if (state.step == num_train_steps) and (rank == 0):
        logger.info("Saving checkpoint...", step=state.step)

        state.save_checkpoint(ckpt_dir=work_dirs.checkpoint, rank=rank)

    logger.finish()


def setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    torch.cuda.set_device(rank)

    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)


def cleanup() -> None:
    dist.destroy_process_group()


def run_mp_training(rank: int, world_size: int, cfg: OmegaConf, port: int) -> None:
    try:
        setup(rank=rank, world_size=world_size, port=port)
        run_train(rank=rank, cfg=cfg)
    finally:
        cleanup()
