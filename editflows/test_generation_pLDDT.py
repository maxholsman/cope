import os
import argparse
import torch
import yaml
from easydict import EasyDict as edict

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.base_models import EditFlow, ProteinEditFlowModel, SMILESEditFlowModel
from model.utils import generate_from_x0
from logic import flow

# tokenizers used in train.py
from transformers import EsmTokenizer, AutoTokenizer, EsmForProteinFolding

from is_peptidomimetic import is_peptidomimetic_not_natural

import pdb

def build_model_and_stuff(cfg, device):
    """
    Rebuild exactly what train.py builds, but we won't set up lightning Trainer.
    Returns:
      editflow_module  (LightningModule)
      source_dist
      (pad_id, bos_id, eos_id)
    """
    if cfg.task == 'protein':
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        vocab_size = tokenizer.vocab_size
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=[0,1,2,3, 24, 25, 26, 27, 28, 29, 30, 31]
        )
        pad_id = 1
        bos_id = 0
        eos_id = 2
        model = ProteinEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    elif cfg.task == 'smiles':
        import selfies as sf
        from smiles_tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
        from smiles_tokenizer.selfies_tokenizers import SelfiesTokenizer
        vocab_size = 587
        tokenizer = SMILES_SPE_Tokenizer(
            '/scratch/pranamlab/tong/cope/editflows/smiles_tokenizer/new_vocab.txt',
            '/scratch/pranamlab/tong/cope/editflows/smiles_tokenizer/new_splits.txt'
        )
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution,
            vocab_size=vocab_size,
            special_token_ids=[0, 1, 2, 3, 4],
        )
        pad_id = 0
        bos_id = 2
        eos_id = 3
        model = SMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    elif cfg.task == 'selfies':
        vocab_size = 44
        tokenizer = SelfiesTokenizer.load("/scratch/pranamlab/tong/data/selfies/28k_mimetics/tokenizer/vocab.json")
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution,
            vocab_size=vocab_size,
            special_token_ids=[0, 1, 2],
        )
        pad_id = 0
        bos_id = 1
        eos_id = 2
        model = SMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    else:
        raise NotImplementedError(f"Unknown task: {cfg.task}")

    eps_id = getattr(cfg.flow, "eps_id", -1)
    path = flow.get_path(
        scheduler_type=cfg.flow.scheduler_type,
        exponent=cfg.flow.exponent,
        eps_id=eps_id,
    )
    loss_fn = flow.get_loss_function(
        loss_function=cfg.flow.loss_function,
        path=path,
    )

    editflow = EditFlow(
        model,
        loss_fn,
        path,
        source_distribution,
        pad_id,
        bos_id,
        eos_id,
        cfg,
    ).to(device)

    return editflow, source_distribution, tokenizer, pad_id, bos_id, eos_id, eps_id


def tokenize_input_str(input_str, cfg, tokenizer, bos_id, eos_id, pad_id, device):
    """
    Turn a user string into x_0 = (1, L) with BOS/EOS and padded.
    This mirrors the training tokenizers as much as we can from here.
    """
    if cfg.task == 'protein':
        toks = tokenizer(input_str, return_tensors='pt')
        ids = toks["input_ids"][0].to(device)
        if ids[0].item() != bos_id:
            ids = torch.cat([torch.tensor([bos_id], device=device), ids], dim=0)
        if ids[-1].item() != eos_id:
            ids = torch.cat([ids, torch.tensor([eos_id], device=device)], dim=0)
        x0 = ids.unsqueeze(0)  # (1, L)
    elif cfg.task == 'smiles':
        ids = tokenizer.encode(input_str)['input_ids']
        ids = torch.tensor(ids, device=device, dtype=torch.long)
        # make sure BOS/EOS
        if ids[0].item() != bos_id:
            ids = torch.cat([torch.tensor([bos_id], device=device), ids], dim=0)
        if ids[-1].item() != eos_id:
            ids = torch.cat([ids, torch.tensor([eos_id], device=device)], dim=0)
        x0 = ids.unsqueeze(0)
    elif cfg.task == 'selfies':
        ids = tokenizer.encode(input_str, already_selfies=False, add_bos_eos=True)['input_ids']
        ids = torch.tensor(ids, device=device, dtype=torch.long)
        if ids[0].item() != bos_id:
            ids = torch.cat([torch.tensor([bos_id], device=device), ids], dim=0)
        if ids[-1].item() != eos_id:
            ids = torch.cat([ids, torch.tensor([eos_id], device=device)], dim=0)
        x0 = ids.unsqueeze(0)
    else:
        raise NotImplementedError

    return x0


def detokenize_output(x, cfg, tokenizer, bos_id, eos_id, pad_id):
    """
    Convert a single generated sequence (1, L) back to string.
    """
    seq = x[0].tolist()
    # strip padding
    seq = [tok for tok in seq if tok != pad_id]
    # strip BOS/EOS
    if len(seq) > 0 and seq[0] == bos_id:
        seq = seq[1:]
    if len(seq) > 0 and seq[-1] == eos_id:
        seq = seq[:-1]

    if cfg.task == 'protein':
        # esm tokenizer has batch_decode
        decoded_seq = tokenizer.batch_decode([seq], skip_special_tokens=True)[0]
        decoded_seq = decoded_seq.replace(' ', '')
        return decoded_seq
    elif cfg.task in ('smiles', 'selfies'):
        return tokenizer.decode(seq)
    else:
        return " ".join(map(str, seq))


def calculate_plddt_from_sequence_string(sequence_string, esmfold_tokenizer, esm_model, device):
    """
    Calculate pLDDT score for a protein sequence string using ESMFold.
    
    Args:
        sequence_string: Protein sequence as string
        esmfold_tokenizer: ESMFold tokenizer
        esm_model: ESMFold model
        device: Device to run on
    
    Returns:
        plddt_score: Mean pLDDT score for the sequence
    """
    tok = esmfold_tokenizer([sequence_string], return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = esm_model(**tok)
        plddt = out.plddt.mean(-1).mean(-1)  # Average across both confidence and sequence length
        return plddt.cpu().item()


def generate_random_sequence(length, alphabet="ACDEFGHIKLMNPQRSTVWY"):
    """
    Generate a random sequence of the given length using the specified alphabet.
    
    Args:
        length: Length of the sequence to generate
        alphabet: String of characters to use for generation
    
    Returns:
        Random sequence string
    """
    return ''.join(np.random.choice(list(alphabet), size=length))
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config_test.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="path to lightning checkpoint (.ckpt)")
    parser.add_argument("--input", type=str, required=True, help="input x_0 as raw string (smiles/protein/selfies)")
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--max-len-cap", type=int, default=None)
    parser.add_argument("--op_temperature", type=float, default=1)
    parser.add_argument("--token_temperature", type=float, default=1)
    parser.add_argument("--num_generations", type=int, default=10, help="number of sequences to generate")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    # Check if task is protein (required for pLDDT calculation)
    if cfg.task != 'protein':
        raise ValueError("pLDDT calculation is only supported for protein task. Current task: {}".format(cfg.task))

    # Load ESMFold model for pLDDT calculation
    print("Loading ESMFold model for pLDDT calculation...")
    esmfold_tokenizer_path = "facebook/esmfold_v1"
    esmfold_tokenizer = AutoTokenizer.from_pretrained(esmfold_tokenizer_path)
    esm_model = EsmForProteinFolding.from_pretrained(esmfold_tokenizer_path, torch_dtype=torch.bfloat16).to(device).eval()
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        esm_model = torch.nn.DataParallel(esm_model)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    else:
        print("Using single GPU")
    
    print("ESMFold model loaded successfully!")

    editflow, source_dist, tokenizer, pad_id, bos_id, eos_id, eps_id = build_model_and_stuff(cfg, device)

    ckpt = torch.load(args.ckpt, map_location=device)
    editflow.load_state_dict(ckpt["state_dict"], strict=False)
    model = editflow.model.to(device)
    model.eval()

    allowed_tokens = torch.tensor(
        [tok for tok in source_dist._allowed_tokens if tok not in (eps_id,)],
        device=device,
        dtype=torch.long,
    )

    # Generate num_generations sequences, each with a new random x0 starting sequence
    print(f"\nGenerating {args.num_generations} sequences (each with a new random starting sequence)...")
    generated_sequences = []
    generated_sequence_strings = []
    natural_AA_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    
    for i in tqdm(range(args.num_generations), desc="Generating sequences"):
        # Generate a new random starting sequence (x0) for each generation
        if cfg.task == 'protein':
            ran_len = random.randint(250, 750)
            random_start_seq = ''.join(np.random.choice(list(natural_AA_alphabet), size=ran_len))
        else:
            # For non-protein tasks, use the provided input or generate random
            if args.input == 'random':
                raise ValueError("Random input generation for non-protein tasks not implemented")
            random_start_seq = args.input
        
        # Tokenize the new random starting sequence
        x0 = tokenize_input_str(random_start_seq, cfg, tokenizer, bos_id, eos_id, pad_id, device)
        
        # Generate from this new x0
        x_gen = generate_from_x0(
            model,
            x0,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            allowed_tokens=allowed_tokens,
            num_steps=args.num_steps,
            max_len_cap=args.max_len_cap,
            op_temperature=args.op_temperature,
            token_temperature=args.token_temperature,
        )
        out_str = detokenize_output(x_gen, cfg, tokenizer, bos_id, eos_id, pad_id)
        generated_sequences.append(x_gen)
        generated_sequence_strings.append(out_str)
    
    print(f"Generated {len(generated_sequence_strings)} sequences")
    print(f"First generated sequence: {generated_sequence_strings[0][:100]}...")

    # Calculate pLDDTs for generated sequences
    print("\nCalculating pLDDT scores for generated sequences...")
    generated_plddt_scores = []
    natural_AA_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    
    for seq_str in tqdm(generated_sequence_strings, desc="Calculating pLDDT for generated sequences"):
        try:
            # Filter sequence to only include valid amino acids
            filtered_seq = ''.join([aa for aa in seq_str if aa in natural_AA_alphabet])
            if len(filtered_seq) > 0:
                plddt_score = calculate_plddt_from_sequence_string(
                    filtered_seq, esmfold_tokenizer, esm_model, device
                )
                generated_plddt_scores.append(plddt_score)
            else:
                print(f"Warning: Generated sequence contains no valid amino acids: {seq_str[:50]}...")
                generated_plddt_scores.append(0.0)
        except Exception as e:
            print(f"Error calculating pLDDT for sequence: {e}")
            generated_plddt_scores.append(0.0)

    print(f"Calculated pLDDT scores for {len(generated_plddt_scores)} generated sequences")

    # Get lengths of generated sequences (after filtering)
    generated_lengths = []
    for seq_str in generated_sequence_strings:
        filtered_seq = ''.join([aa for aa in seq_str if aa in natural_AA_alphabet])
        generated_lengths.append(len(filtered_seq))

    # Generate random sequences matching the lengths
    print(f"\nGenerating {args.num_generations} random sequences matching lengths...")
    random_sequence_strings = []
    for length in generated_lengths:
        if length > 0:
            random_seq = generate_random_sequence(length, natural_AA_alphabet)
            random_sequence_strings.append(random_seq)
        else:
            # If length is 0, generate a random length sequence
            random_len = random.randint(250, 750)
            random_seq = generate_random_sequence(random_len, natural_AA_alphabet)
            random_sequence_strings.append(random_seq)

    # Calculate pLDDTs for random sequences
    print("Calculating pLDDT scores for random sequences...")
    random_plddt_scores = []
    
    for seq_str in tqdm(random_sequence_strings, desc="Calculating pLDDT for random sequences"):
        try:
            plddt_score = calculate_plddt_from_sequence_string(
                seq_str, esmfold_tokenizer, esm_model, device
            )
            random_plddt_scores.append(plddt_score)
        except Exception as e:
            print(f"Error calculating pLDDT for random sequence: {e}")
            random_plddt_scores.append(0.0)

    print(f"Calculated pLDDT scores for {len(random_plddt_scores)} random sequences")

    # Print statistics
    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    print(f"Generated sequences:")
    print(f"  Count: {len(generated_plddt_scores)}")
    print(f"  Mean pLDDT: {np.mean(generated_plddt_scores):.2f}")
    print(f"  Std pLDDT: {np.std(generated_plddt_scores):.2f}")
    print(f"  Min pLDDT: {np.min(generated_plddt_scores):.2f}")
    print(f"  Max pLDDT: {np.max(generated_plddt_scores):.2f}")
    print(f"\nRandom sequences:")
    print(f"  Count: {len(random_plddt_scores)}")
    print(f"  Mean pLDDT: {np.mean(random_plddt_scores):.2f}")
    print(f"  Std pLDDT: {np.std(random_plddt_scores):.2f}")
    print(f"  Min pLDDT: {np.min(random_plddt_scores):.2f}")
    print(f"  Max pLDDT: {np.max(random_plddt_scores):.2f}")

    # Plot histogram
    print("\nCreating histogram plot...")
    plt.figure(figsize=(12, 8))
    
    plt.hist(generated_plddt_scores, bins=15, alpha=0.6, label='Generated Sequences', color='blue', density=True)
    plt.hist(random_plddt_scores, bins=15, alpha=0.6, label='Random Sequences', color='orange', density=True)
    
    plt.xlabel('pLDDT Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('pLDDT Score Distributions: Generated vs Random Sequences', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
#     stats_text = f"""Statistics:
# Generated: μ={np.mean(generated_plddt_scores):.2f}, σ={np.std(generated_plddt_scores):.2f}
# Random: μ={np.mean(random_plddt_scores):.2f}, σ={np.std(random_plddt_scores):.2f}

# N = {len(generated_plddt_scores)} sequences each"""
    
#     plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
#              fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plddt_histogram_generated_vs_random.png', dpi=300, bbox_inches='tight')
    print("Histogram saved as 'plddt_histogram_generated_vs_random.png'")
    plt.show()

    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)



if __name__ == "__main__":
    main()
