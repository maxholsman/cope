import os
import sys
import argparse
import torch
import yaml
from easydict import EasyDict as edict
import random
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import matplotlib.pyplot as plt
from collections import defaultdict
from datasets import load_from_disk
from difflib import SequenceMatcher

from model.base_models import EditFlow, ProteinEditFlowModel, SMILESEditFlowModel, ReparameterizedProteinEditFlowModel
from model.utils import generate_from_x0, generate_from_x0_ctmc
from logic import flow
from transformers import EsmTokenizer, AutoTokenizer, EsmForProteinFolding

import pdb

def build_model_and_stuff(cfg, device):
    """
    Rebuild exactly what train.py builds, but we won't set up lightning Trainer.
    Returns:
      editflow_module  (LightningModule)
      source_dist
      (pad_id, bos_id, eos_id)
      is_reparameterized (bool)
    """
    # Initialize is_reparameterized (only protein task supports reparameterized models currently)
    is_reparameterized = False
    
    if cfg.task == 'protein':
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        vocab_size = tokenizer.vocab_size
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=[0,1,2,3, 24, 25, 26, 27, 28, 29, 30, 31]
        )
        pad_id = 1
        bos_id = 0
        eos_id = 2
        # Check if reparameterized model should be used
        is_reparameterized = getattr(cfg.training, "reparameterize", False)
        if is_reparameterized:
            model = ReparameterizedProteinEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
        else:
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
        tokenizer = SelfiesTokenizer.load("/usr/xtmp/mth45/Documents/programmable_biology_group/cope/data/28k_mimetics/tokenizer/vocab.json")
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

    # Verify model type matches is_reparameterized flag (for robustness)
    if isinstance(model, ReparameterizedProteinEditFlowModel) != is_reparameterized:
        # If there's a mismatch, trust the instance type
        is_reparameterized = isinstance(model, ReparameterizedProteinEditFlowModel)

    return editflow, source_distribution, tokenizer, pad_id, bos_id, eos_id, eps_id, is_reparameterized


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


def generate_random_sequence(length, alphabet="ACDEFGHIKLMNPQRSTVWY"):
    """Generate a random sequence of given length using the amino acid alphabet."""
    return ''.join(np.random.choice(list(alphabet), size=length))


def load_uniprot_sequences_from_fasta(fasta_path):
    """
    Load UniProt sequences from a FASTA file and organize by length for efficient lookup.
    Returns a dictionary mapping length -> list of sequences.
    """
    print(f"Loading UniProt sequences from FASTA file: {fasta_path}...")
    sequences_by_length = defaultdict(list)
    
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Loading UniProt"):
        seq_str = str(record.seq)
        # Only keep sequences with valid amino acids
        if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
            sequences_by_length[len(seq_str)].append(seq_str)
    
    print(f"Loaded sequences for {len(sequences_by_length)} different lengths")
    print(f"Total sequences: {sum(len(seqs) for seqs in sequences_by_length.values())}")
    
    return sequences_by_length


def load_exclusion_set(exclusion_path, tokenizer=None, bos_id=None, eos_id=None, pad_id=None):
    """
    Load exclusion sequences from either a FASTA file or a pre-batched HuggingFace dataset.
    Returns a set of sequence strings for fast lookup.
    
    Args:
        exclusion_path: Path to FASTA file or dataset directory
        tokenizer: Tokenizer (required if loading from dataset)
        bos_id: BOS token ID (required if loading from dataset)
        eos_id: EOS token ID (required if loading from dataset)
        pad_id: PAD token ID (required if loading from dataset)
    
    Returns:
        Set of sequence strings to exclude
    """
    exclusion_set = set()
    
    # Determine if it's a FASTA file or a dataset directory
    if os.path.isfile(exclusion_path) and exclusion_path.endswith(('.fasta', '.fa', '.fas')):
        # Load from FASTA file
        print(f"Loading exclusion sequences from FASTA file: {exclusion_path}...")
        for record in tqdm(SeqIO.parse(exclusion_path, "fasta"), desc="Loading exclusion sequences"):
            seq_str = str(record.seq)
            # Only keep sequences with valid amino acids
            if all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                exclusion_set.add(seq_str)
    elif os.path.isdir(exclusion_path):
        # Load from pre-batched dataset
        if tokenizer is None or bos_id is None or eos_id is None or pad_id is None:
            raise ValueError("tokenizer, bos_id, eos_id, and pad_id must be provided when loading exclusion sequences from a dataset")
        
        print(f"Loading exclusion sequences from dataset: {exclusion_path}...")
        dataset = load_from_disk(exclusion_path)
        print(f"Dataset loaded with {len(dataset)} items")
        
        # Process each item in the dataset
        for item in tqdm(dataset, desc="Processing exclusion dataset"):
            # Extract input_ids (tokenized sequence)
            input_ids = item['input_ids']
            
            # Handle both list and tensor formats
            if isinstance(input_ids, list):
                # If it's a list of lists (batched), process each sequence
                if len(input_ids) > 0 and isinstance(input_ids[0], list):
                    for seq_ids in input_ids:
                        seq_str = detokenize_sequence(seq_ids, tokenizer, bos_id, eos_id, pad_id)
                        if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                            exclusion_set.add(seq_str)
                else:
                    # Single sequence as list
                    seq_str = detokenize_sequence(input_ids, tokenizer, bos_id, eos_id, pad_id)
                    if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                        exclusion_set.add(seq_str)
            elif isinstance(input_ids, torch.Tensor):
                # If it's a tensor, convert to list first
                input_ids = input_ids.tolist()
                if len(input_ids) > 0 and isinstance(input_ids[0], list):
                    for seq_ids in input_ids:
                        seq_str = detokenize_sequence(seq_ids, tokenizer, bos_id, eos_id, pad_id)
                        if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                            exclusion_set.add(seq_str)
                else:
                    seq_str = detokenize_sequence(input_ids, tokenizer, bos_id, eos_id, pad_id)
                    if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                        exclusion_set.add(seq_str)
            else:
                # Try to convert to list
                try:
                    input_ids = list(input_ids)
                    seq_str = detokenize_sequence(input_ids, tokenizer, bos_id, eos_id, pad_id)
                    if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                        exclusion_set.add(seq_str)
                except Exception as e:
                    print(f"Warning: Could not process input_ids: {e}")
                    continue
    else:
        raise ValueError(
            f"Invalid path for exclusion sequences: {exclusion_path}\n"
            f"Must be either:\n"
            f"  1. A FASTA file (.fasta, .fa, .fas)\n"
            f"  2. A directory containing a HuggingFace dataset"
        )
    
    print(f"Loaded {len(exclusion_set)} exclusion sequences")
    
    return exclusion_set


def load_uniprot_sequences_from_dataset(dataset_path, tokenizer, bos_id, eos_id, pad_id):
    """
    Load UniProt sequences from a pre-batched HuggingFace dataset and organize by length.
    The dataset should contain 'input_ids' which are tokenized sequences.
    Returns a dictionary mapping length -> list of sequences.
    """
    print(f"Loading UniProt sequences from dataset: {dataset_path}...")
    sequences_by_length = defaultdict(list)
    
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded with {len(dataset)} items")
    
    # Process each item in the dataset
    for item in tqdm(dataset, desc="Processing dataset"):
        # Extract input_ids (tokenized sequence)
        input_ids = item['input_ids']
        
        # Handle both list and tensor formats
        if isinstance(input_ids, list):
            # If it's a list of lists (batched), process each sequence
            if len(input_ids) > 0 and isinstance(input_ids[0], list):
                for seq_ids in input_ids:
                    seq_str = detokenize_sequence(seq_ids, tokenizer, bos_id, eos_id, pad_id)
                    if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                        sequences_by_length[len(seq_str)].append(seq_str)
            else:
                # Single sequence as list
                seq_str = detokenize_sequence(input_ids, tokenizer, bos_id, eos_id, pad_id)
                if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                    sequences_by_length[len(seq_str)].append(seq_str)
        elif isinstance(input_ids, torch.Tensor):
            # If it's a tensor, convert to list first
            input_ids = input_ids.tolist()
            if len(input_ids) > 0 and isinstance(input_ids[0], list):
                for seq_ids in input_ids:
                    seq_str = detokenize_sequence(seq_ids, tokenizer, bos_id, eos_id, pad_id)
                    if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                        sequences_by_length[len(seq_str)].append(seq_str)
            else:
                seq_str = detokenize_sequence(input_ids, tokenizer, bos_id, eos_id, pad_id)
                if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                    sequences_by_length[len(seq_str)].append(seq_str)
        else:
            # Try to convert to list
            try:
                input_ids = list(input_ids)
                seq_str = detokenize_sequence(input_ids, tokenizer, bos_id, eos_id, pad_id)
                if seq_str and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq_str):
                    sequences_by_length[len(seq_str)].append(seq_str)
            except Exception as e:
                print(f"Warning: Could not process input_ids: {e}")
                continue
    
    print(f"Loaded sequences for {len(sequences_by_length)} different lengths")
    print(f"Total sequences: {sum(len(seqs) for seqs in sequences_by_length.values())}")
    
    return sequences_by_length


def detokenize_sequence(token_ids, tokenizer, bos_id, eos_id, pad_id):
    """
    Convert token IDs to a protein sequence string.
    Strips BOS, EOS, and padding tokens.
    """
    # Convert to list if needed
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    # Strip padding, BOS, and EOS
    seq = [tok for tok in token_ids if tok != pad_id]
    if len(seq) > 0 and seq[0] == bos_id:
        seq = seq[1:]
    if len(seq) > 0 and seq[-1] == eos_id:
        seq = seq[:-1]
    
    # Decode using tokenizer
    try:
        decoded_seq = tokenizer.batch_decode([seq], skip_special_tokens=True)[0]
        decoded_seq = decoded_seq.replace(' ', '')
        return decoded_seq
    except Exception as e:
        print(f"Warning: Error detokenizing sequence: {e}")
        return None


def find_uniprot_sequence_of_length(sequences_by_length, target_length, max_length_diff=None):
    """
    Find a UniProt sequence of the given length.
    If exact match not found, find the closest length.
    
    Args:
        sequences_by_length: Dictionary mapping length -> list of sequences
        target_length: Desired sequence length
        max_length_diff: Maximum allowed length difference. If None, allows any difference.
                        If specified, returns None if closest match exceeds this difference.
    
    Returns:
        A sequence string, or None if no suitable match found
    """
    # Try exact match first
    if target_length in sequences_by_length and len(sequences_by_length[target_length]) > 0:
        return random.choice(sequences_by_length[target_length])
    
    # Find closest length
    available_lengths = sorted(sequences_by_length.keys())
    if not available_lengths:
        return None
    
    # Find closest length
    closest_length = min(available_lengths, key=lambda x: abs(x - target_length))
    length_diff = abs(closest_length - target_length)
    
    # If max_length_diff is specified and difference is too large, return None
    if max_length_diff is not None and length_diff > max_length_diff:
        return None
    
    if len(sequences_by_length[closest_length]) > 0:
        return random.choice(sequences_by_length[closest_length])
    
    return None


def calculate_edit_statistics(original_seq, generated_seq):
    """
    Calculate edit statistics between original and generated sequences.
    
    Returns:
        dict with keys: num_deletions, num_substitutions, num_insertions, 
                       total_edits, num_unedited
    """
    matcher = SequenceMatcher(None, original_seq, generated_seq)
    
    num_deletions = 0
    num_substitutions = 0
    num_insertions = 0
    num_unedited = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Matching segment
            num_unedited += (i2 - i1)
        elif tag == 'delete':
            # Deletion: characters in original but not in generated
            num_deletions += (i2 - i1)
        elif tag == 'insert':
            # Insertion: characters in generated but not in original
            num_insertions += (j2 - j1)
        elif tag == 'replace':
            # Substitution: different characters at same position
            # Count as substitution (one per position in original)
            num_substitutions += (i2 - i1)
    
    total_edits = num_deletions + num_substitutions + num_insertions
    
    return {
        'num_deletions': num_deletions,
        'num_substitutions': num_substitutions,
        'num_insertions': num_insertions,
        'total_edits': total_edits,
        'num_unedited': num_unedited
    }


def make_random_edits(sequence, num_edits, alphabet="ACDEFGHIKLMNPQRSTVWY"):
    """
    Make num_edits random edits to a sequence.
    Each edit can be:
    - Substitution: replace a character at a random position
    - Deletion: delete a character at a random position
    - Insertion: insert a random character at a random position
    
    Returns:
        Modified sequence string
    """
    seq_list = list(sequence)
    
    for _ in range(num_edits):
        if len(seq_list) == 0:
            # If sequence is empty, can only insert
            edit_type = 'insert'
        else:
            edit_type = random.choice(['substitute', 'delete', 'insert'])
        
        if edit_type == 'substitute':
            # Substitute a random character
            pos = random.randint(0, len(seq_list) - 1)
            seq_list[pos] = random.choice(alphabet)
        elif edit_type == 'delete':
            # Delete a random character
            pos = random.randint(0, len(seq_list) - 1)
            seq_list.pop(pos)
        elif edit_type == 'insert':
            # Insert a random character at a random position
            pos = random.randint(0, len(seq_list))
            seq_list.insert(pos, random.choice(alphabet))
    
    return ''.join(seq_list)


def make_random_edits_to_match_length(sequence, target_length, num_substitutions, alphabet="ACDEFGHIKLMNPQRSTVWY", debug=False):
    """
    Make random edits to a sequence to match a target length, then apply substitutions.
    This ensures the final length exactly matches the target, regardless of how SequenceMatcher
    counts edits.
    
    Args:
        sequence: Original sequence string
        target_length: Desired final length
        num_substitutions: Number of substitutions to apply (doesn't change length)
        alphabet: Alphabet to use for edits
        debug: If True, print debugging information
    
    Returns:
        Modified sequence string with length matching target_length
    """
    seq_list = list(sequence)
    original_len = len(seq_list)
    length_diff = target_length - original_len
    
    if debug:
        print(f"    [Debug] Target length: {target_length}, original: {original_len}, diff: {length_diff}")
    
    # First, adjust length to match target
    if length_diff > 0:
        # Need to add characters (insertions)
        if debug:
            print(f"    [Debug] Applying {length_diff} insertions")
        for _ in range(length_diff):
            pos = random.randint(0, len(seq_list))
            seq_list.insert(pos, random.choice(alphabet))
    elif length_diff < 0:
        # Need to remove characters (deletions)
        num_deletions = abs(length_diff)
        if debug:
            print(f"    [Debug] Applying {num_deletions} deletions")
        if len(seq_list) > 0:
            deletion_positions = sorted(random.sample(range(len(seq_list)), min(num_deletions, len(seq_list))), reverse=True)
            for pos in deletion_positions:
                seq_list.pop(pos)
    
    # Now apply substitutions (doesn't change length)
    actual_substitutions = 0
    if num_substitutions > 0 and len(seq_list) > 0:
        max_substitutions = min(num_substitutions, len(seq_list))
        substitution_positions = random.sample(range(len(seq_list)), max_substitutions)
        if debug:
            print(f"    [Debug] Applying {len(substitution_positions)} substitutions at positions: {substitution_positions[:10]}{'...' if len(substitution_positions) > 10 else ''}")
        for pos in substitution_positions:
            current_char = seq_list[pos]
            available_chars = [c for c in alphabet if c != current_char]
            if available_chars:
                seq_list[pos] = random.choice(available_chars)
            else:
                seq_list[pos] = random.choice(alphabet)
            actual_substitutions += 1
    
    final_len = len(seq_list)
    
    if debug:
        print(f"    [Debug] Summary: Applied length adjustment (diff={length_diff}), subs={actual_substitutions} "
              f"(original_len={original_len}, final_len={final_len}, target_len={target_length})")
    
    # Verify length matches (should always be true, but good to check)
    if final_len != target_length:
        raise ValueError(f"Length mismatch: expected {target_length}, got {final_len}")
    
    return ''.join(seq_list)


def calculate_plddt_from_sequence_string(sequence_string, esmfold_tokenizer, esm_model, device):
    """
    Calculate pLDDT score for a sequence string using ESMFold.
    Based on test_data_pLDDT.py
    """
    try:
        tok = esmfold_tokenizer([sequence_string], return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = esm_model(**tok)
            plddt = out.plddt.mean(-1).mean(-1)  # Average across both confidence and sequence length
            # Handle scalar or tensor output
            if plddt.dim() > 0:
                plddt = plddt[0]  # Take first element if batch dimension exists
            return plddt.cpu().item()
    except Exception as e:
        print(f"Error calculating pLDDT for sequence (length {len(sequence_string)}): {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config_test.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="path to lightning checkpoint (.ckpt)")
    parser.add_argument("--input", type=str, required=True, help="input x_0 as raw string (smiles/protein/selfies)")
    parser.add_argument("--num-steps", type=int, nargs='+', default=[32],
                       help="List of num-steps values to run. Each value will generate a separate plot.")
    parser.add_argument("--max-len-cap", type=int, default=None)
    parser.add_argument("--op_temperature", type=float, default=1)
    parser.add_argument("--token_temperature", type=float, default=1)
    parser.add_argument("--pos_temperature", type=float, default=1.0,
                       help="Temperature for sampling position in reparameterized models. Only used when model is reparameterized and --sample-type is 'reparameterized'")
    parser.add_argument("--n", type=int, default=100, help="Number of sequences to generate")
    parser.add_argument("--true-sequences", type=str, default=None,
                       help="Path to true protein sequences: either a FASTA file (.fasta) or a pre-batched dataset directory")
    parser.add_argument("--uniprot-fasta", type=str, 
                       default="/usr/xtmp/mth45/Documents/programmable_biology_group/cope/data/uniprot_sprot.fasta",
                       help="[DEPRECATED] Path to UniProt FASTA file. Use --true-sequences instead.")
    parser.add_argument("--output-dir", type=str, default="./plddt_analysis_output", 
                       help="Directory to save output plots")
    parser.add_argument("--num-bins", type=int, default=50, 
                       help="Number of bins to use for the histogram")
    parser.add_argument("--max-length-diff", type=int, default=0,
                       help="Maximum allowed length difference between generated and true sequences. "
                            "Sequences with larger differences will be filtered out. Default: 0 (exact match only)")
    parser.add_argument("--skip-length-matching", action="store_true",
                       help="Skip length matching requirement. Will use random true sequences regardless of length. "
                            "Useful for comparing overall pLDDT distributions rather than length-matched comparisons.")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Optional run name to append to plot filenames and display in plot titles")
    parser.add_argument("--max-input-length", type=int, default=None,
                       help="Maximum length of real x0 sequences when using --input 'real'. Sequences longer than this will be filtered out.")
    parser.add_argument("--exclusion-sequences", type=str, default=None,
                       help="Path to exclusion sequences: either a FASTA file (.fasta) or a pre-batched dataset directory. "
                            "Sequences from this source will be excluded from real_sequences_pool. "
                            "Useful for excluding training data when using --input 'real'.")
    parser.add_argument("--sample-type", type=str, default="reparameterized", choices=["vanilla", "reparameterized"],
                       help="Sampling type for reparameterized models: 'vanilla' uses convert_to_vanilla_outputs=True, "
                            "'reparameterized' uses convert_to_vanilla_outputs=False. Default: 'reparameterized'")
    
    # Use mutually exclusive group for CTMC flag
    ctmc_group = parser.add_mutually_exclusive_group()
    ctmc_group.add_argument("--use-ctmc", action="store_true", default=None,
                           help="Use generate_from_x0_ctmc for generation")
    ctmc_group.add_argument("--no-use-ctmc", dest="use_ctmc", action="store_false",
                           help="Disable CTMC generation and use generate_from_x0 instead (default behavior)")

    args = parser.parse_args()
    
    # Set default to False (no CTMC) if neither flag was provided
    if args.use_ctmc is None:
        args.use_ctmc = False

    # Extract run name from checkpoint path
    # Expected format: .../outputs/{run_name}/checkpoint/last.ckpt
    ckpt_path = args.ckpt
    ckpt_dir = os.path.dirname(ckpt_path)  # Get directory containing the .ckpt file
    run_name = os.path.basename(os.path.dirname(ckpt_dir))  # Get parent directory name (the run name)
    
    # Modify output directory to include run name as subfolder
    base_output_dir = args.output_dir
    args.output_dir = os.path.join(base_output_dir, run_name)
    print(f"Extracted run name: {run_name}")
    print(f"Output directory: {args.output_dir}")

    # Determine device assignment
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            # Use separate GPUs: EditFlow on GPU 0, ESMFold on GPU 1
            editflow_device = torch.device("cuda:0")
            esmfold_device = torch.device("cuda:1")
            print(f"Using 2 GPUs: EditFlow on {editflow_device}, ESMFold on {esmfold_device}")
        else:
            # Use single GPU for both
            editflow_device = torch.device("cuda:0")
            esmfold_device = torch.device("cuda:0")
            print(f"Using single GPU: {editflow_device}")
    else:
        editflow_device = torch.device("cpu")
        esmfold_device = torch.device("cpu")
        print("Using CPU")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    # Only support protein task for now
    if cfg.task != 'protein':
        raise ValueError("This script currently only supports the 'protein' task")

    # Build model
    print("Building EditFlow model...")
    editflow, source_dist, tokenizer, pad_id, bos_id, eos_id, eps_id, is_reparameterized = build_model_and_stuff(cfg, editflow_device)
    
    if is_reparameterized:
        print("Detected reparameterized model - using reparameterized generation logic")
    else:
        print("Detected base model - using base generation logic")
    
    # Determine convert_to_vanilla_outputs based on sample_type and model type
    if is_reparameterized:
        if args.sample_type == "vanilla":
            convert_to_vanilla_outputs = True
            print(f"Using vanilla sampling (convert_to_vanilla_outputs=True)")
        elif args.sample_type == "reparameterized":
            convert_to_vanilla_outputs = False
            print(f"Using reparameterized sampling (convert_to_vanilla_outputs=False)")
        else:
            raise ValueError(f"Invalid sample_type: {args.sample_type}. Must be 'vanilla' or 'reparameterized'")
    else:
        # For non-reparameterized models, this flag doesn't matter, but set to False for consistency
        convert_to_vanilla_outputs = False
        print(f"Note: --sample-type is ignored for non-reparameterized models")

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=editflow_device)
    try:
        editflow.load_state_dict(ckpt["state_dict"], strict=True)
    except RuntimeError as e:
        print(f"Warning: {e}")
        editflow.load_state_dict(ckpt["state_dict"], strict=False)

    model = editflow.model.to(editflow_device)
    model.eval()

    # Load ESMFold for pLDDT calculation
    print("Loading ESMFold model for pLDDT calculation...")
    esmfold_tokenizer_path = "facebook/esmfold_v1"
    esmfold_tokenizer = AutoTokenizer.from_pretrained(esmfold_tokenizer_path)
    esm_model = EsmForProteinFolding.from_pretrained(esmfold_tokenizer_path, torch_dtype=torch.bfloat16).to(esmfold_device).eval()
    
    print("ESMFold model loaded successfully!")

    # Load true sequences early if using 'real' input mode
    real_sequences_pool = None
    if args.input == 'real':
        print(f"\nLoading true sequences for 'real' input mode...")
        # Determine which source to use: --true-sequences takes precedence over --uniprot-fasta
        true_sequences_path = args.true_sequences if args.true_sequences else args.uniprot_fasta
        
        if true_sequences_path is None:
            raise ValueError("--true-sequences must be provided when using --input 'real'")
        
        # Determine if it's a FASTA file or a dataset directory
        if os.path.isfile(true_sequences_path) and true_sequences_path.endswith(('.fasta', '.fa', '.fas')):
            # Load from FASTA file
            sequences_by_length = load_uniprot_sequences_from_fasta(true_sequences_path)
        elif os.path.isdir(true_sequences_path):
            # Load from pre-batched dataset
            sequences_by_length = load_uniprot_sequences_from_dataset(
                true_sequences_path, tokenizer, bos_id, eos_id, pad_id
            )
        else:
            raise ValueError(
                f"Invalid path for true sequences: {true_sequences_path}\n"
                f"Must be either:\n"
                f"  1. A FASTA file (.fasta, .fa, .fas)\n"
                f"  2. A directory containing a HuggingFace dataset"
            )
        
        # Flatten sequences_by_length into a single pool
        real_sequences_pool = []
        for length_group in sequences_by_length.values():
            real_sequences_pool.extend(length_group)
        
        # Filter by max_input_length if specified
        if args.max_input_length is not None:
            original_count = len(real_sequences_pool)
            real_sequences_pool = [seq for seq in real_sequences_pool if len(seq) <= args.max_input_length]
            filtered_count = len(real_sequences_pool)
            if filtered_count < original_count:
                print(f"Filtered sequences by max_input_length={args.max_input_length}: "
                      f"{original_count} -> {filtered_count} sequences")
        
        # Remove exclusion sequences if exclusion path is provided
        if args.exclusion_sequences is not None:
            exclusion_set = load_exclusion_set(
                args.exclusion_sequences, 
                tokenizer=tokenizer, 
                bos_id=bos_id, 
                eos_id=eos_id, 
                pad_id=pad_id
            )
            original_count = len(real_sequences_pool)
            real_sequences_pool = [seq for seq in real_sequences_pool if seq not in exclusion_set]
            filtered_count = len(real_sequences_pool)
            excluded_count = original_count - filtered_count
            
            if excluded_count > 0:
                print(f"Excluded {excluded_count} sequences found in exclusion set: "
                      f"{original_count} -> {filtered_count} sequences")
            else:
                print(f"No sequences from real_sequences_pool were found in exclusion set")
        
        if len(real_sequences_pool) == 0:
            error_msg = "No true sequences found in dataset!"
            if args.max_input_length is not None:
                error_msg += f" (after filtering by max_input_length={args.max_input_length})"
            if args.exclusion_sequences is not None:
                error_msg += f" (after excluding sequences from {args.exclusion_sequences})"
            raise ValueError(error_msg)
        
        print(f"Loaded {len(real_sequences_pool)} true sequences for 'real' input mode")

    # Process each num_steps value
    num_steps_list = args.num_steps if isinstance(args.num_steps, list) else [args.num_steps]
    print(f"\nProcessing {len(num_steps_list)} num_steps values: {num_steps_list}")
    
    for num_steps in num_steps_list:
        print(f"\n{'='*80}")
        print(f"Processing num_steps = {num_steps}")
        print(f"{'='*80}")
        
        # Generate n sequences first (before loading UniProt)
        print(f"\nGenerating {args.n} sequences with num_steps={num_steps}...")
        generated_sequences = []
        random_sequences = []
        initial_x0_lengths = []  # store the (core) length of the initial x0 for each generated sequence
        initial_x0_sequences = []  # store the original x0 sequences (as strings) for edit analysis
        ctmc_total_edits = []  # track total edits for CTMC generation (x0 -> x1)
        percent_original_unedited = []  # Percentage of original sequence that remains unedited (for real mode)
        percent_generated_unedited = []  # Percentage of generated sequence that comes from original (for real mode)
        num_unedited_list = []  # Actual number of unedited tokens for each sequence (for real mode)
        use_real_mode = (args.input == 'real')
        
        allowed_tokens = torch.tensor(
            [tok for tok in source_dist._allowed_tokens if tok not in (eps_id,)],
            device=editflow_device,
            dtype=torch.long,
        )

        # Get scale_size from config (same as used during training)
        scale_size = getattr(cfg.model, "scale_size", 2.0)
        
        # Prepare input sequence(s)
        for i in tqdm(range(args.n), desc="Generating sequences"):
            # If input is 'random', sample x0 the same way as during training
            if args.input == 'random' and cfg.task == 'protein':
                # During training, x0 is sampled from x1 using sample_x0_from_x1
                # x0 core length is in [0, scale_size * x1_valid_len]
                # To get x0 with desired length range, we create a dummy x1 and sample from it
                
                # Target x0 core length range (excluding BOS/EOS)
                # Using same range as before: [0, 250]
                target_max_x0_len = 250
                
                # Create dummy x1 with appropriate length
                # x1_valid_len should be >= target_max_x0_len / scale_size to allow x0 up to target_max_x0_len
                dummy_x1_valid_len = max(1, int(np.ceil(target_max_x0_len / scale_size)))
                
                # Create dummy x1: [BOS] + valid_len tokens + [EOS]
                # Sample tokens uniformly from allowed_tokens
                dummy_core_tokens = allowed_tokens[torch.randint(0, len(allowed_tokens), (dummy_x1_valid_len,), device=editflow_device)]
                dummy_x1 = torch.cat([
                    torch.tensor([bos_id], device=editflow_device, dtype=torch.long),
                    dummy_core_tokens,
                    torch.tensor([eos_id], device=editflow_device, dtype=torch.long),
                ], dim=0).unsqueeze(0)  # (1, L)
                
                # Sample x0 from dummy_x1 using the same method as training
                with torch.no_grad():
                    x0 = source_dist.sample_x0_from_x1(
                        dummy_x1,
                        pad_id=pad_id,
                        allowed_tokens=allowed_tokens,
                        scale_size=scale_size,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        eps_id=eps_id
                    )  # (1, L0)
            elif args.input == 'real' and cfg.task == 'protein':
                # Use a random real protein sequence as x0
                if real_sequences_pool is None or len(real_sequences_pool) == 0:
                    raise ValueError("No real sequences available for 'real' input mode")
                input_seq = random.choice(real_sequences_pool)
                x0 = tokenize_input_str(input_seq, cfg, tokenizer, bos_id, eos_id, pad_id, editflow_device)
            else:
                # Use provided input string
                input_seq = args.input
                x0 = tokenize_input_str(input_seq, cfg, tokenizer, bos_id, eos_id, pad_id, editflow_device)
            
            # Record initial x0 core length (excluding BOS/EOS and PAD), to compare against generated length
            x0_core_mask = (x0[0] != pad_id) & (x0[0] != bos_id) & (x0[0] != eos_id)
            initial_x0_len = int(x0_core_mask.sum().item())
            initial_x0_lengths.append(initial_x0_len)
            
            # Store the original x0 sequence as a string for edit analysis
            x0_seq = detokenize_output(x0, cfg, tokenizer, bos_id, eos_id, pad_id)
            initial_x0_sequences.append(x0_seq)

            # Generate sequence
            if args.use_ctmc:
                x_gen = generate_from_x0_ctmc(
                    model,
                    x0,
                    pad_id=pad_id,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    allowed_tokens=allowed_tokens,
                    num_steps=num_steps,
                    max_len_cap=args.max_len_cap,
                    op_temperature=args.op_temperature,
                    token_temperature=args.token_temperature,
                    pos_temperature=args.pos_temperature,
                    is_reparameterized=is_reparameterized,
                    convert_to_vanilla_outputs=convert_to_vanilla_outputs,
                )
            else:
                x_gen = generate_from_x0(
                    model,
                    x0,
                    pad_id=pad_id,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    allowed_tokens=allowed_tokens,
                    num_steps=num_steps,
                    max_len_cap=args.max_len_cap,
                    op_temperature=args.op_temperature,
                    token_temperature=args.token_temperature,
                    pos_temperature=args.pos_temperature,
                    is_reparameterized=is_reparameterized,
                    convert_to_vanilla_outputs=convert_to_vanilla_outputs,
                )
            
            # Detokenize
            gen_seq = detokenize_output(x_gen, cfg, tokenizer, bos_id, eos_id, pad_id)
            generated_sequences.append(gen_seq)
            
            # Track edits for CTMC generation (x0 -> x1)
            if args.use_ctmc:
                edit_stats = calculate_edit_statistics(x0_seq, gen_seq)
                ctmc_total_edits.append(edit_stats['total_edits'])
            
            # Create random sequence or random baseline
            if use_real_mode:
                # For 'real' mode: create random baseline with matching length and substitution count
                # Calculate edit statistics from generated sequence (for substitution count)
                edit_stats = calculate_edit_statistics(x0_seq, gen_seq)
                seq_idx = len(generated_sequences)
                print(f"  [Debug] Seq {seq_idx}: Creating random baseline - "
                      f"target_len={len(gen_seq)}, subs={edit_stats['num_substitutions']}")
                # Create random baseline with same length and substitution count
                # This ensures length always matches, regardless of SequenceMatcher's edit counting
                random_seq = make_random_edits_to_match_length(
                    x0_seq,
                    len(gen_seq),  # Target length is the generated sequence length
                    edit_stats['num_substitutions'],
                    debug=True
                )
            else:
                # For other modes: create completely random sequence of equal length
                random_seq = generate_random_sequence(len(gen_seq))
            random_sequences.append(random_seq)

        # For debugging/analysis: print edit statistics for each sequence
        if use_real_mode:
            # In 'real' mode, print edit statistics for both generated and random baseline sequences
            # Also track unedited percentages (lists already initialized above)
            print("Edit statistics for generated sequences (x0_len, gen_len, deletions, substitutions, insertions, total_edits, unedited, %_original_unedited, %_generated_unedited):")
            for i, gen_seq in enumerate(generated_sequences):
                gen_len = len(gen_seq)
                x0_len = initial_x0_lengths[i] if i < len(initial_x0_lengths) else None
                x0_seq = initial_x0_sequences[i] if i < len(initial_x0_sequences) else None
                
                if x0_seq is not None:
                    edit_stats = calculate_edit_statistics(x0_seq, gen_seq)
                    
                    # Calculate unedited percentages
                    if len(x0_seq) > 0:
                        pct_orig = (edit_stats['num_unedited'] / len(x0_seq)) * 100.0
                    else:
                        pct_orig = 0.0
                    
                    if len(gen_seq) > 0:
                        pct_gen = (edit_stats['num_unedited'] / len(gen_seq)) * 100.0
                    else:
                        pct_gen = 0.0
                    
                    percent_original_unedited.append(pct_orig)
                    percent_generated_unedited.append(pct_gen)
                    num_unedited_list.append(edit_stats['num_unedited'])
                    
                    print(f"  Generated Seq {i}: (x0_len={x0_len}, gen_len={gen_len}, "
                          f"dels={edit_stats['num_deletions']}, subs={edit_stats['num_substitutions']}, "
                          f"ins={edit_stats['num_insertions']}, total_edits={edit_stats['total_edits']}, "
                          f"unedited={edit_stats['num_unedited']}, "
                          f"%_orig_unedited={pct_orig:.2f}%, %_gen_unedited={pct_gen:.2f}%)")
                else:
                    print(f"  Generated Seq {i}: (x0_len={x0_len}, gen_len={gen_len}, edit_stats=N/A)")
            
            # Print summary statistics for unedited percentages
            if len(percent_original_unedited) > 0:
                print("\n" + "="*60)
                print("Unedited Percentage Statistics (Real Input Mode):")
                print("="*60)
                print(f"Percentage of original sequence that remains unedited:")
                print(f"  Mean: {np.mean(percent_original_unedited):.2f}%")
                print(f"  Median: {np.median(percent_original_unedited):.2f}%")
                print(f"  Std: {np.std(percent_original_unedited):.2f}%")
                print(f"  Min: {np.min(percent_original_unedited):.2f}%, Max: {np.max(percent_original_unedited):.2f}%")
                print(f"\nPercentage of generated sequence that comes from original (unedited):")
                print(f"  Mean: {np.mean(percent_generated_unedited):.2f}%")
                print(f"  Median: {np.median(percent_generated_unedited):.2f}%")
                print(f"  Std: {np.std(percent_generated_unedited):.2f}%")
                print(f"  Min: {np.min(percent_generated_unedited):.2f}%, Max: {np.max(percent_generated_unedited):.2f}%")
                print(f"  N = {len(percent_generated_unedited)} sequences")
                print("="*60)
            
            print("\nRandom baseline sequences (x0_len, baseline_len):")
            for i, random_seq in enumerate(random_sequences):
                baseline_len = len(random_seq)
                x0_len = initial_x0_lengths[i] if i < len(initial_x0_lengths) else None
                print(f"  Baseline Seq {i}: (x0_len={x0_len}, baseline_len={baseline_len})")
        else:
            # For other modes, only print edit statistics for generated sequences
            print("Edit statistics for each sample (x0_len, gen_len, deletions, substitutions, insertions, total_edits, unedited):")
            for i, gen_seq in enumerate(generated_sequences):
                gen_len = len(gen_seq)
                x0_len = initial_x0_lengths[i] if i < len(initial_x0_lengths) else None
                x0_seq = initial_x0_sequences[i] if i < len(initial_x0_sequences) else None
                
                if x0_seq is not None:
                    edit_stats = calculate_edit_statistics(x0_seq, gen_seq)
                    print(f"  Seq {i}: (x0_len={x0_len}, gen_len={gen_len}, "
                          f"dels={edit_stats['num_deletions']}, subs={edit_stats['num_substitutions']}, "
                          f"ins={edit_stats['num_insertions']}, total_edits={edit_stats['total_edits']}, "
                          f"unedited={edit_stats['num_unedited']})")
                else:
                    print(f"  Seq {i}: (x0_len={x0_len}, gen_len={gen_len}, edit_stats=N/A)")
        
        # Handle true sequences differently for 'real' mode vs other modes
        if use_real_mode:
            # In 'real' mode, use the initial x0 sequences as the "real sequences" for comparison
            print(f"\nUsing 'real' input mode - x0 sequences are the real protein sequences")
            true_sequences = initial_x0_sequences.copy()
            print(f"Using {len(true_sequences)} real sequences (x0 inputs) for comparison")
        else:
            # For other modes, load true protein sequences for the lengths we actually need
            print(f"\nLoading true protein sequences for required lengths...")
            required_lengths = set(len(seq) for seq in generated_sequences)
            print(f"Required generated lengths (unique): {sorted(required_lengths)}")   
        
            # Determine which source to use: --true-sequences takes precedence over --uniprot-fasta
            true_sequences_path = args.true_sequences if args.true_sequences else args.uniprot_fasta
        
            # Determine if it's a FASTA file or a dataset directory
            if os.path.isfile(true_sequences_path) and true_sequences_path.endswith(('.fasta', '.fa', '.fas')):
                # Load from FASTA file
                sequences_by_length = load_uniprot_sequences_from_fasta(true_sequences_path)
            elif os.path.isdir(true_sequences_path):
                # Load from pre-batched dataset
                sequences_by_length = load_uniprot_sequences_from_dataset(
                    true_sequences_path, tokenizer, bos_id, eos_id, pad_id
                )
            else:
                raise ValueError(
                    f"Invalid path for true sequences: {true_sequences_path}\n"
                    f"Must be either:\n"
                    f"  1. A FASTA file (.fasta, .fa, .fas)\n"
                    f"  2. A directory containing a HuggingFace dataset"
                )
        
            # Find true protein sequences
            if args.skip_length_matching:
                print(f"\nSkipping length matching - using random true sequences regardless of length...")
                # Just randomly sample true sequences from the dataset
                all_true_sequences = []
                for length_group in sequences_by_length.values():
                    all_true_sequences.extend(length_group)
        
                if len(all_true_sequences) == 0:
                    raise ValueError("No true sequences found in dataset!")
        
                print(f"Found {len(all_true_sequences)} total true sequences in dataset")
                # Sample random true sequences (with replacement if needed)
                true_sequences = [random.choice(all_true_sequences) for _ in range(len(generated_sequences))]
        
                print(f"Using {len(true_sequences)} randomly sampled true sequences (lengths may vary)")
            else:
                print(f"\nFinding true protein sequences of matching lengths...")
                print(f"Using max_length_diff={args.max_length_diff} (0 = exact match only)")
        
                # Filter sequences to only keep those with good matches
                filtered_generated = []
                filtered_random = []
                filtered_true = []
                filtered_x0_lengths = []
                filtered_x0_sequences = []
                filtered_ctmc_edits = []
                skipped_count = 0
        
                for i, gen_seq in enumerate(tqdm(generated_sequences, desc="Matching true sequences")):
                    true_seq = find_uniprot_sequence_of_length(
                        sequences_by_length, 
                        len(gen_seq), 
                        max_length_diff=args.max_length_diff if args.max_length_diff >= 0 else None
                    )
            
                    if true_seq is None:
                        skipped_count += 1
                        if skipped_count <= 10:  # Only print first 10 warnings
                            print(f"Warning: Skipping generated sequence of length {len(gen_seq)} - no suitable true sequence found")
                        continue
            
                    # Verify lengths match (should always be true now, but double-check)
                    if len(gen_seq) != len(true_seq) and args.max_length_diff == 0:
                        skipped_count += 1
                        if skipped_count <= 10:
                            print(f"Warning: Skipping generated sequence of length {len(gen_seq)} - true sequence length {len(true_seq)} doesn't match")
                        continue
            
                    filtered_generated.append(gen_seq)
                    filtered_random.append(random_sequences[i])
                    filtered_true.append(true_seq)
                    # Also filter the corresponding x0 data
                    if i < len(initial_x0_lengths):
                        filtered_x0_lengths.append(initial_x0_lengths[i])
                    if i < len(initial_x0_sequences):
                        filtered_x0_sequences.append(initial_x0_sequences[i])
                    # Also filter CTMC edit counts if available
                    if args.use_ctmc and i < len(ctmc_total_edits):
                        filtered_ctmc_edits.append(ctmc_total_edits[i])
        
                if skipped_count > 10:
                    print(f"... and {skipped_count - 10} more sequences skipped")
        
                print(f"\nFiltered to {len(filtered_generated)} sequences with matching true sequences (skipped {skipped_count})")
        
                if len(filtered_generated) == 0:
                    raise ValueError(
                        "No sequences with matching true sequences found! "
                        "Try increasing --max-length-diff, using --skip-length-matching, or check that your true sequences dataset contains sequences of similar lengths."
                    )
        
                # Update the lists to use filtered versions
                generated_sequences = filtered_generated
                random_sequences = filtered_random
                true_sequences = filtered_true
                initial_x0_lengths = filtered_x0_lengths
                initial_x0_sequences = filtered_x0_sequences
                if args.use_ctmc:
                    ctmc_total_edits = filtered_ctmc_edits

        print(f"\nFinal dataset: {len(generated_sequences)} sequences")
        print(f"Average generated length: {np.mean([len(s) for s in generated_sequences]):.2f}")
        print(f"Average true length: {np.mean([len(s) for s in true_sequences]):.2f}")

        # Verify that all three sets have sequences of equal length at each index (only if length matching is enabled and not in real mode)
        if not use_real_mode and not args.skip_length_matching:
            print("\nVerifying sequence lengths match across all three sets...")
            length_mismatches = []
            for i in range(len(generated_sequences)):
                gen_len = len(generated_sequences[i])
                rand_len = len(random_sequences[i])
                true_len = len(true_sequences[i])
        
                if not (gen_len == rand_len == true_len):
                    length_mismatches.append({
                        'index': i,
                        'generated': gen_len,
                        'random': rand_len,
                        'true': true_len
                    })
    
            if length_mismatches:
                print(f"ERROR: Found {len(length_mismatches)} length mismatches after filtering:")
                for mismatch in length_mismatches[:10]:  # Show first 10
                    print(f"  Index {mismatch['index']}: Generated={mismatch['generated']}, Random={mismatch['random']}, True={mismatch['true']}")
                if len(length_mismatches) > 10:
                    print(f"  ... and {len(length_mismatches) - 10} more mismatches")
                raise ValueError(f"Sequence length mismatch detected! All sequences at index i must have the same length.")
            else:
                print(f"✓ All {len(generated_sequences)} sequences have matching lengths across all three sets")
        elif not use_real_mode:
            print("\n⚠ Length matching disabled - comparing overall pLDDT distributions (lengths may differ)")

        # Calculate pLDDT scores
        if use_real_mode:
            # In 'real' mode: calculate pLDDT for real x0 sequences, generated sequences, and random baseline
            print("\nCalculating pLDDT scores for real x0 sequences...")
            real_x0_plddt = []
            for seq in tqdm(initial_x0_sequences, desc="Real x0 sequences"):
                score = calculate_plddt_from_sequence_string(seq, esmfold_tokenizer, esm_model, esmfold_device)
                if score is not None:
                    real_x0_plddt.append(score)
    
            print("\nCalculating pLDDT scores for generated sequences...")
            generated_plddt = []
            for seq in tqdm(generated_sequences, desc="Generated sequences"):
                score = calculate_plddt_from_sequence_string(seq, esmfold_tokenizer, esm_model, esmfold_device)
                if score is not None:
                    generated_plddt.append(score)

            print("\nCalculating pLDDT scores for random baseline sequences...")
            random_baseline_plddt = []
            for seq in tqdm(random_sequences, desc="Random baseline sequences"):
                score = calculate_plddt_from_sequence_string(seq, esmfold_tokenizer, esm_model, esmfold_device)
                if score is not None:
                    random_baseline_plddt.append(score)
    
            # Ensure all three lists have the same length (take minimum)
            min_len = min(len(real_x0_plddt), len(generated_plddt), len(random_baseline_plddt))
            if min_len == 0:
                raise ValueError("No valid pLDDT scores calculated! Check your sequences and ESMFold model.")
    
            real_x0_plddt = real_x0_plddt[:min_len]
            generated_plddt = generated_plddt[:min_len]
            random_baseline_plddt = random_baseline_plddt[:min_len]
    
            print(f"\nUsing {min_len} sequences for each group (after filtering failed calculations)")

            # Print statistics
            print("\n" + "="*60)
            print("pLDDT Statistics (Real Input Mode):")
            print("="*60)
            print(f"Real x0 sequences:      N={len(real_x0_plddt)}, μ={np.mean(real_x0_plddt):.2f}, σ={np.std(real_x0_plddt):.2f}")
            print(f"Generated sequences:    N={len(generated_plddt)}, μ={np.mean(generated_plddt):.2f}, σ={np.std(generated_plddt):.2f}")
            print(f"Random baseline:        N={len(random_baseline_plddt)}, μ={np.mean(random_baseline_plddt):.2f}, σ={np.std(random_baseline_plddt):.2f}")
            print("="*60)
        else:
            # For other modes: calculate pLDDT for generated, random, and true sequences
            print("\nCalculating pLDDT scores for generated sequences...")
            generated_plddt = []
            for seq in tqdm(generated_sequences, desc="Generated sequences"):
                score = calculate_plddt_from_sequence_string(seq, esmfold_tokenizer, esm_model, esmfold_device)
                if score is not None:
                    generated_plddt.append(score)

            print("\nCalculating pLDDT scores for random sequences...")
            random_plddt = []
            for seq in tqdm(random_sequences, desc="Random sequences"):
                score = calculate_plddt_from_sequence_string(seq, esmfold_tokenizer, esm_model, esmfold_device)
                if score is not None:
                    random_plddt.append(score)

            print("\nCalculating pLDDT scores for true protein sequences...")
            true_plddt = []
            for seq in tqdm(true_sequences, desc="True sequences"):
                score = calculate_plddt_from_sequence_string(seq, esmfold_tokenizer, esm_model, esmfold_device)
                if score is not None:
                    true_plddt.append(score)
    
            # Ensure all three lists have the same length (take minimum)
            min_len = min(len(generated_plddt), len(random_plddt), len(true_plddt))
            if min_len == 0:
                raise ValueError("No valid pLDDT scores calculated! Check your sequences and ESMFold model.")
    
            generated_plddt = generated_plddt[:min_len]
            random_plddt = random_plddt[:min_len]
            true_plddt = true_plddt[:min_len]
    
            print(f"\nUsing {min_len} sequences for each group (after filtering failed calculations)")

            # Print statistics
            print("\n" + "="*60)
            print("pLDDT Statistics:")
            print("="*60)
            print(f"Generated sequences: N={len(generated_plddt)}, μ={np.mean(generated_plddt):.2f}, σ={np.std(generated_plddt):.2f}")
            print(f"Random sequences:    N={len(random_plddt)}, μ={np.mean(random_plddt):.2f}, σ={np.std(random_plddt):.2f}")
            print(f"True sequences:      N={len(true_plddt)}, μ={np.mean(true_plddt):.2f}, σ={np.std(true_plddt):.2f}")
            print("="*60)

        # Plot histograms
        print("\nCreating histogram plots...")
        plt.figure(figsize=(14, 8))

        if use_real_mode:
            # In 'real' mode: plot real x0, generated, and random baseline
            # Compute bin edges based on the combined range of all three datasets
            all_scores = real_x0_plddt + generated_plddt + random_baseline_plddt
            min_score = min(all_scores)
            max_score = max(all_scores)
            bin_edges = np.linspace(min_score, max_score, args.num_bins + 1)

            # Plot histograms for all three groups using the same bin edges
            plt.hist(real_x0_plddt, bins=bin_edges, alpha=0.6, label='Real x0', color='green', density=True)
            plt.hist(generated_plddt, bins=bin_edges, alpha=0.6, label='Generated', color='blue', density=True)
            plt.hist(random_baseline_plddt, bins=bin_edges, alpha=0.6, label='Random Baseline', color='red', density=True)

            plt.xlabel('pLDDT Score', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            title = 'pLDDT Score Distributions: Real x0 vs Generated vs Random Baseline'
            current_run_name = args.run_name if args.run_name else run_name
            if current_run_name:
                title += f' ({current_run_name}_{num_steps}_steps)'
            plt.title(title, fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)

            # Add statistics text box
            stats_text = f"""Statistics:
    Real x0:         μ={np.mean(real_x0_plddt):.2f}, σ={np.std(real_x0_plddt):.2f}
    Generated:       μ={np.mean(generated_plddt):.2f}, σ={np.std(generated_plddt):.2f}
    Random Baseline: μ={np.mean(random_baseline_plddt):.2f}, σ={np.std(random_baseline_plddt):.2f}

    N = {len(generated_plddt)} sequences each"""

            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                     fontsize=10)

            plt.tight_layout()
            filename = 'plddt_histogram_comparison'
            current_run_name = args.run_name if args.run_name else run_name
            if current_run_name:
                filename += f'_{current_run_name}_{num_steps}_steps'
            filename += '.png'
            output_path = os.path.join(args.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved histogram to {output_path}")
            plt.close()

            # Create box plot
            plt.figure(figsize=(12, 6))
            data_for_box = [real_x0_plddt, generated_plddt, random_baseline_plddt]
            labels = ['Real x0', 'Generated', 'Random Baseline']

            plt.boxplot(data_for_box, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))

            plt.ylabel('pLDDT Score', fontsize=12)
            title = 'pLDDT Score Distribution Comparison (Box Plot) - Real Input Mode'
            current_run_name = args.run_name if args.run_name else run_name
            if current_run_name:
                title += f' ({current_run_name}_{num_steps}_steps)'
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = 'plddt_boxplot_comparison'
            current_run_name = args.run_name if args.run_name else run_name
            if current_run_name:
                filename += f'_{current_run_name}_{num_steps}_steps'
            filename += '.png'
            output_path = os.path.join(args.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved boxplot to {output_path}")
            plt.close()
        else:
            # For other modes: plot generated, random, and true
            # Compute bin edges based on the combined range of all three datasets
            all_scores = generated_plddt + random_plddt + true_plddt
            min_score = min(all_scores)
            max_score = max(all_scores)
            bin_edges = np.linspace(min_score, max_score, args.num_bins + 1)

            # Plot histograms for all three groups using the same bin edges
            plt.hist(generated_plddt, bins=bin_edges, alpha=0.6, label='Generated', color='blue', density=True)
            plt.hist(random_plddt, bins=bin_edges, alpha=0.6, label='Random', color='red', density=True)
            plt.hist(true_plddt, bins=bin_edges, alpha=0.6, label='True', color='green', density=True)

            plt.xlabel('pLDDT Score', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            title = 'pLDDT Score Distributions: Generated vs Random vs True Sequences'
            current_run_name = args.run_name if args.run_name else run_name
            if current_run_name:
                title += f' ({current_run_name}_{num_steps}_steps)'
            plt.title(title, fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)

            # Add statistics text box
            stats_text = f"""Statistics:
    Generated: μ={np.mean(generated_plddt):.2f}, σ={np.std(generated_plddt):.2f}
    Random:    μ={np.mean(random_plddt):.2f}, σ={np.std(random_plddt):.2f}
    True:      μ={np.mean(true_plddt):.2f}, σ={np.std(true_plddt):.2f}

    N = {len(generated_plddt)} sequences each"""

            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                     fontsize=10)

            plt.tight_layout()
            filename = 'plddt_histogram_comparison'
            current_run_name = args.run_name if args.run_name else run_name
            if current_run_name:
                filename += f'_{current_run_name}_{num_steps}_steps'
            filename += '.png'
            output_path = os.path.join(args.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved histogram to {output_path}")
            plt.close()

            # Create box plot
            plt.figure(figsize=(12, 6))
            data_for_box = [generated_plddt, random_plddt, true_plddt]
            labels = ['Generated', 'Random', 'True']

            plt.boxplot(data_for_box, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))

            plt.ylabel('pLDDT Score', fontsize=12)
            title = 'pLDDT Score Distribution Comparison (Box Plot)'
            current_run_name = args.run_name if args.run_name else run_name
            if current_run_name:
                title += f' ({current_run_name}_{num_steps}_steps)'
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = 'plddt_boxplot_comparison'
            current_run_name = args.run_name if args.run_name else run_name
            if current_run_name:
                filename += f'_{current_run_name}_{num_steps}_steps'
            filename += '.png'
            output_path = os.path.join(args.output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved boxplot to {output_path}")
            plt.close()

        # Print completion message for this num_steps iteration
        print(f"\n{'='*80}")
        print(f"✅ Completed num_steps = {num_steps}")
        print(f"📊 Figures saved to: {args.output_dir}")
        
        # Print CTMC edit statistics if CTMC was used
        if args.use_ctmc and len(ctmc_total_edits) > 0:
            total_edits = sum(ctmc_total_edits)
            avg_edits = np.mean(ctmc_total_edits)
            median_edits = np.median(ctmc_total_edits)
            min_edits = min(ctmc_total_edits)
            max_edits = max(ctmc_total_edits)
            std_edits = np.std(ctmc_total_edits)
            print(f"\n📝 CTMC Edit Statistics (x0 -> x1):")
            print(f"   Total edits across all sequences: {total_edits:,}")
            print(f"   Average edits per sequence: {avg_edits:.2f}")
            print(f"   Median edits per sequence: {median_edits:.2f}")
            print(f"   Min edits: {min_edits}, Max edits: {max_edits}")
            print(f"   Standard deviation: {std_edits:.2f}")
            print(f"   Number of sequences: {len(ctmc_total_edits)}")
        
        # Print unedited percentage statistics if in real mode
        if use_real_mode and len(percent_original_unedited) > 0:
            avg_orig = np.mean(percent_original_unedited)
            median_orig = np.median(percent_original_unedited)
            min_orig = np.min(percent_original_unedited)
            max_orig = np.max(percent_original_unedited)
            avg_gen = np.mean(percent_generated_unedited)
            median_gen = np.median(percent_generated_unedited)
            min_gen = np.min(percent_generated_unedited)
            max_gen = np.max(percent_generated_unedited)
            print(f"\n📊 Unedited Percentage Statistics (Real Input Mode):")
            print(f"   Percentage of original sequence that remains unedited:")
            print(f"      Average: {avg_orig:.2f}%, Median: {median_orig:.2f}%")
            print(f"      Min: {min_orig:.2f}%, Max: {max_orig:.2f}%")
            print(f"   Percentage of generated sequence that comes from original (unedited):")
            print(f"      Average: {avg_gen:.2f}%, Median: {median_gen:.2f}%")
            print(f"      Min: {min_gen:.2f}%, Max: {max_gen:.2f}%")
            print(f"   Number of sequences: {len(percent_original_unedited)}")
        
        print(f"{'='*80}\n")

    print("\n🎉 Analysis complete for all num_steps values!")
    print(f"📊 All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

