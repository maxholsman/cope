# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
from contextlib import nullcontext
from typing import Optional

import torch
from flow_matching.loss import MixturePathGeneralizedKL, EditFlowsLoss
from flow_matching.path import ProbPath
from omegaconf.dictconfig import DictConfig
from torch import nn, Tensor
from torch.cuda.amp import GradScaler

from torch.utils.data import DataLoader
from utils.logging import TrainLogger

from .flow import SourceDistribution
from .state import TrainState

from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch import Tensor

@dataclass
class EFTargetsRagged:
    # x_t (ε stripped) and per-token targets
    x_t_list: List[Tensor]                 # list[(n_i,)]
    need_delete_list: List[Tensor]         # list[(n_i,)]   bool
    need_substitute_list: List[Tensor]     # list[(n_i,)]   bool
    sub_target_list: List[Tensor]          # list[(n_i,)]   long
    # insertion events (variable count)
    ins_slot_idx_list: List[Tensor]        # list[(K_i,)]   long in [0..n_i]
    ins_target_list: List[Tensor]          # list[(K_i,)]   long

@torch.no_grad()
def build_alignment(
    x0_list: List[Tensor],        # list[(M_i,)]  source tokens
    x1_list: List[Tensor],        # list[(N_i,)]  target tokens
    eps_id: int,
    strategy: str = "random_50_50",
    generator: torch.Generator | None = None,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Produce aligned Z-space sequences with ε for each sample independently.
    Returns lists: z0_list[i], z1_list[i] both shape (T_i,).
    """
    assert len(x0_list) == len(x1_list), "x0_list and x1_list must have same batch length"
    B = len(x0_list)
    z0_list: List[Tensor] = []
    z1_list: List[Tensor] = []

    for b in range(B):
        src = x0_list[b]  # (M,)
        tgt = x1_list[b]  # (N,)
        device = src.device
        dtype  = src.dtype

        M = int(src.numel())
        N = int(tgt.numel())

        if strategy == "random_50_50":
            # choose substitutions among x0 positions (at most N)
            k_sub = min(M // 2, N)
            if k_sub > 0:
                sub_idx = torch.randperm(M, device=device, generator=generator)[:k_sub]
                sub_mask = torch.zeros(M, dtype=torch.bool, device=device)
                sub_mask[sub_idx] = True
            else:
                sub_mask = torch.zeros(M, dtype=torch.bool, device=device)
        else:
            raise ValueError(f"Unknown alignment strategy: {strategy}")

        z0_parts = []
        z1_parts = []
        j = 0  # pointer in x1

        # walk x0 in order: either (x0_i, x1_j) for subs, else (x0_i, ε) for deletes
        for i in range(M):
            if sub_mask[i] and j < N:
                z0_parts.append(src[i:i+1])             # substitution
                z1_parts.append(tgt[j:j+1])
                j += 1
            else:
                z0_parts.append(src[i:i+1])             # deletion
                z1_parts.append(torch.tensor([eps_id], device=device, dtype=dtype))

        # remaining x1 are insertions: (ε, x1_j)
        while j < N:
            z0_parts.append(torch.tensor([eps_id], device=device, dtype=dtype))
            z1_parts.append(tgt[j:j+1])
            j += 1

        z0_b = torch.cat(z0_parts, dim=0)
        z1_b = torch.cat(z1_parts, dim=0)
        assert z0_b.shape == z1_b.shape
        z0_list.append(z0_b)
        z1_list.append(z1_b)

    return z0_list, z1_list


@torch.no_grad()
def aligned_to_ef_targets_ragged(
    z_t_list: List[Tensor],    # list[(T_i,)]
    z1_list:  List[Tensor],    # list[(T_i,)]
    eps_id: int,
) -> EFTargetsRagged:
    """
    Convert aligned (z_t, z_1) to:
      - x_t (ε stripped)
      - per-token delete/substitute masks and substitution targets
      - insertion events: (slot_idx, ins_target) pairs (K_i events per sample)
    All ragged; no padding.
    """
    assert len(z_t_list) == len(z1_list)
    B = len(z_t_list)

    x_t_list: List[Tensor] = []
    need_del_list: List[Tensor] = []
    need_sub_list: List[Tensor] = []
    sub_tgt_list: List[Tensor] = []
    ins_slot_idx_list: List[Tensor] = []
    ins_tgt_list: List[Tensor] = []

    for i in range(B):
        zt = z_t_list[i]
        z1 = z1_list[i]
        device = zt.device
        dtype  = zt.dtype
        T = zt.numel()
        assert z1.numel() == T

        # first pass: compute x_t and build a map from aligned pos -> token index in x_t
        x_tokens = []
        token_index_of_pos = torch.full((T,), -1, device=device, dtype=torch.long)  # -1 for ε
        n = 0
        for k in range(T):
            if zt[k].item() != eps_id:
                token_index_of_pos[k] = n
                x_tokens.append(zt[k:k+1])
                n += 1
        x_t = torch.cat(x_tokens, dim=0) if n > 0 else torch.empty(0, dtype=dtype, device=device)

        # init per-token arrays
        need_del   = torch.zeros(n, dtype=torch.bool, device=device)
        need_sub   = torch.zeros(n, dtype=torch.bool, device=device)
        sub_target = torch.zeros(n, dtype=dtype,        device=device)

        # insertion events (variable count)
        ins_slot_idx = []
        ins_targets  = []

        # second pass: fill masks/targets using running count to determine slot indices
        # running token count equals "how many non-ε tokens seen so far"
        seen = 0
        for k in range(T):
            a = zt[k].item()
            b = z1[k].item()

            if a != eps_id and b == eps_id:
                # delete at current token index
                tok_idx = token_index_of_pos[k].item()
                if tok_idx >= 0:
                    need_del[tok_idx] = True
                seen += 1  # we saw a token in z_t
            elif a != eps_id and b != eps_id:
                tok_idx = token_index_of_pos[k].item()
                if tok_idx >= 0 and a != b:
                    need_sub[tok_idx] = True
                    sub_target[tok_idx] = z1[k]
                seen += 1
            elif a == eps_id and b != eps_id:
                # insertion event goes to slot 'seen' (in [0..n])
                ins_slot_idx.append(seen)
                ins_targets.append(z1[k])
            else:
                # (ε, ε) — should be rare in our construction; ignore
                pass

        # finalize tensors
        ins_slot_idx_t = torch.tensor(ins_slot_idx, dtype=torch.long, device=device) if ins_slot_idx else torch.empty(0, dtype=torch.long, device=device)
        ins_targets_t  = torch.stack(ins_targets, dim=0) if ins_targets else torch.empty(0, dtype=dtype, device=device)

        x_t_list.append(x_t)
        need_del_list.append(need_del)
        need_sub_list.append(need_sub)
        sub_tgt_list.append(sub_target)
        ins_slot_idx_list.append(ins_slot_idx_t)
        ins_tgt_list.append(ins_targets_t)

    return EFTargetsRagged(
        x_t_list=x_t_list,
        need_delete_list=need_del_list,
        need_substitute_list=need_sub_list,
        sub_target_list=sub_tgt_list,
        ins_slot_idx_list=ins_slot_idx_list,
        ins_target_list=ins_tgt_list,
    )


def _get_lr(lr: float, step: int, warmup: int, n_iters: int, eta_min_ratio: float):
    if step < warmup:
        # Linear warmup
        return lr * (step / warmup)
    else:
        # Cosine annealing
        total_steps = n_iters
        eta_min = eta_min_ratio * lr
        cosine_decay = 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup))
        )
        return eta_min + (lr - eta_min) * cosine_decay


def optimization_step(
    state: TrainState,
    scaler: GradScaler,
    loss: Tensor,
    optim_params: DictConfig,
    logger: TrainLogger,
) -> None:
    scaler.scale(loss).backward()
    scaler.unscale_(state.optimizer)

    lr = _get_lr(
        lr=optim_params.lr,
        step=state.step,
        warmup=optim_params.warmup,
        n_iters=optim_params.n_iters,
        eta_min_ratio=optim_params.eta_min_ratio,
    )

    # Update learning rate in optimizer
    for g in state.optimizer.param_groups:
        g["lr"] = lr

    if state.step % optim_params.log_lr_every == 0:
        logger.log_lr(value=lr, step=state.step)

    if optim_params.grad_clip >= 0:
        torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), max_norm=optim_params.grad_clip
        )

    scaler.step(state.optimizer)
    scaler.update()

    state.optimizer.zero_grad()

def step(
    state: TrainState,
    loss_fn: nn.Module,                 # EditFlowsLoss (ragged-aware)
    path: ProbPath,                     # EditFlowsPathAdapter (exposes .scheduler)
    scaler: GradScaler,
    iterator: DataLoader,
    device: torch.device,
    source_distribution: SourceDistribution,
    logger: TrainLogger,
    training: bool,
    optim_params: Optional[DictConfig] = None,
    time_epsilon: float = 0.0,          # unused (Option A)
) -> Tensor:
    assert (training and (optim_params is not None)) or (not training)
    state.train() if training else state.eval()

    batch = next(iterator)
    x_1 = [t.to(device) for t in batch["input_ids"]]   # list[Tensor] for ragged, as *target*
    B = len(x_1)
    # print(f"sample_x_1: {x_1[0]}")

    # === Source & time ===
    with torch.no_grad():
        # Get eps_id from path to exclude from sampling
        eps_id = getattr(path, "eps_id", -1)
        
        # Source: empty / empirical / uniform (your choice)
        # Get allowed tokens by excluding eps_id from default allowed tokens (vocab minus special tokens)
        # We exclude eps_id because it's a special alignment token (typically -1) and should not be sampled
        # If source_distribution has _allowed_tokens, use those; otherwise sample from all vocab
        if hasattr(source_distribution, "_allowed_tokens"):
            # Filter out eps_id from allowed tokens (it shouldn't be there anyway, but filter for safety)
            allowed_tokens = [tok for tok in source_distribution._allowed_tokens if tok != eps_id]
        else:
            # Fallback: compute allowed tokens manually if needed
            allowed_tokens = None
        
        x_0 = source_distribution.sample_like(x_1, allowed_tokens=allowed_tokens)  # list[Tensor] for ragged
        # print(f"sample_x_0: {x_0[0]}")
        # Option A: t ~ Uniform(0,1)
        t = torch.rand(B, device=device)
        # print(f"sample_t: {t}")
        # Precompute κ and weight (clamp κ only here)
        sched = path.scheduler(t)
        kappa = sched.alpha_t        # κ(t)
        kappa_dot = sched.d_alpha_t  # dκ/dt
        kappa_eps = 1e-6
        kappa_safe = torch.clamp(kappa, max=1.0 - kappa_eps)
        precomputed_weight = kappa_dot / (1.0 - kappa_safe)     # (B,)

    # === Build alignments (ragged in Z) ===
    # eps_id already defined above from path
    with torch.no_grad():
        # Pad ragged sequences to batched tensors for alignment
        # Find max length for padding
        
        z0_list, z1_list = build_alignment(
            x0_list=x_0,                  # <-- lists, not padded tensors
            x1_list=x_1,
            eps_id=eps_id,
            strategy=getattr(getattr(logger, "cfg", {}), "flow", {}).get("alignment", "random_50_50")
        )

        # === Sample z_t in Z via the adapter path (ragged-aware) ===
        # Should return a struct with .zt_list (list[Tensor]) or equivalent
        zt_sample = path.sample(z0=z0_list, z1=z1_list, t=t)
        # print(f"zt_sample: {zt_sample.zt_list[0].tolist()}")
        z_t_list = getattr(zt_sample, "zt_list", None)
        if z_t_list is None:
            # if your adapter returns a flat tensor, convert to list by splitting with original aligned lengths
            z_t_list = zt_sample.x_t_list  # prefer adapter to expose ragged directly

        # === Convert aligned (z_t, z_1) → ragged x_t and EF targets/masks (ragged) ===
        targets = aligned_to_ef_targets_ragged(z_t_list=z_t_list, z1_list=z1_list, eps_id=eps_id)

        # print(f"x0_list:                        {x_0[0].tolist()}")
        # print(f"x_t_list:                       {zt_sample.x_t_list[0].tolist()}")
        # print(f"x1_list:                        {x_1[0].tolist()}")
        # print(f"z0_list:                        {z0_list[0].tolist()}")
        # print(f"z_t_list:                       {z_t_list[0].tolist()}")
        # print(f"z1_list:                        {z1_list[0].tolist()}")
        # print(f"targets.need_delete_list:       {targets.need_delete_list[0].to(torch.int).tolist()}")
        # print(f"targets.need_substitute_list:   {targets.need_substitute_list[0].to(torch.int).tolist()}")
        # print(f"targets.sub_target_list:        {targets.sub_target_list[0]}")
        # print(f"targets.ins_slot_idx_list:      {targets.ins_slot_idx_list[0]}")
        # print(f"targets.ins_target_list:        {targets.ins_target_list[0]}")

    # === Forward pass (ragged) ===
    ctx = nullcontext() if training else torch.no_grad()
    with ctx:
        # IMPORTANT: model expects ragged x_t (list of LongTensors)
        lam_ins_list, logits_ins_list, lam_del_list, lam_sub_list, logits_sub_list = state.model(
            x_t=targets.x_t_list,  # ragged
            time=t,                # (B,)
        )

        # Remove singleton batch dimensions from model outputs (model returns (1, n) but loss expects (n,))
        lam_ins_list = [t.squeeze(0) if t.dim() > 1 and t.size(0) == 1 else t for t in lam_ins_list]
        logits_ins_list = [t.squeeze(0) if t.dim() > 2 and t.size(0) == 1 else t for t in logits_ins_list]
        lam_del_list = [t.squeeze(0) if t.dim() > 1 and t.size(0) == 1 else t for t in lam_del_list]
        lam_sub_list = [t.squeeze(0) if t.dim() > 1 and t.size(0) == 1 else t for t in lam_sub_list]
        logits_sub_list = [t.squeeze(0) if t.dim() > 2 and t.size(0) == 1 else t for t in logits_sub_list]

        # === Loss (ragged-aware) ===
        loss = loss_fn(
            lam_ins=lam_ins_list,
            lam_del=lam_del_list,
            lam_sub=lam_sub_list,
            logits_ins=logits_ins_list,
            logits_sub=logits_sub_list,
            need_delete=targets.need_delete_list,
            need_substitute=targets.need_substitute_list,
            ins_slot_idx=targets.ins_slot_idx_list,
            ins_target=targets.ins_target_list,
            sub_target=targets.sub_target_list,
            t=t,
            precomputed_weight=precomputed_weight,   # (B,)
        )

    if training:
        optimization_step(
            state=state,
            loss=loss,
            scaler=scaler,
            optim_params=optim_params,
            logger=logger,
        )

    return loss.detach()


def step_old(
    state: TrainState,
    loss_fn: nn.Module,
    path: ProbPath,
    scaler: GradScaler,
    iterator: DataLoader,
    device: torch.device,
    source_distribution: SourceDistribution,
    logger: TrainLogger,
    training: bool,
    optim_params: Optional[DictConfig] = None,
    time_epsilon: float = 0.0,
) -> Tensor:
    assert (training and (optim_params is not None)) or (not training)

    if training:
        state.train()
    else:
        state.eval()

    x_1 = next(iterator)["input_ids"].to(device)

    # Sample from path
    with torch.no_grad():
        x_0 = source_distribution.sample_like(x_1)
        # determine z_0 and z_1 given x_0 and x_1
        t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1) # need to update this to use z_0 and z_1

    # Forward and compute loss
    ctx = nullcontext() if training else torch.no_grad()

    with ctx:
        logits = state.model(x_t=path_sample.x_t, time=path_sample.t)

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
        elif isinstance(loss_fn, MixturePathGeneralizedKL):
            loss = loss_fn(
                logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t
            ).mean()
        else:
            raise ValueError("Invalid loss function")

    # Optimization step (only if training=true)
    if training:
        optimization_step(
            state=state,
            loss=loss,
            scaler=scaler,
            optim_params=optim_params,
            logger=logger,
        )

    return loss.detach()
