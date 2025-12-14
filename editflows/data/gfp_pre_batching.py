import os
from datasets import Dataset, DatasetDict

# import selfies as sf
# from rdkit import Chem
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')

import sys

from transformers import EsmTokenizer

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BATCH_SIZE = 1
SOURCE_DATASET = 'gfp'
MAX_LENGTH = 1500
MIN_LENGTH = None
MAX_DATA_SIZE = None
NUM_REPEATS = None
PAD_ID = 1

# Input directory containing train.fasta, val.fasta, test.fasta
if SOURCE_DATASET == 'gfp':
    INPUT_DIR = '/usr/xtmp/mth45/Documents/programmable_biology_group/cope/data/gfp/fpbase_pfamPF01353'  # directory with train.fasta, val.fasta, test.fasta
    TRAIN_FASTA = os.path.join(INPUT_DIR, 'train.fasta')
    VAL_FASTA = os.path.join(INPUT_DIR, 'val.fasta')
    TEST_FASTA = os.path.join(INPUT_DIR, 'test.fasta')
else:

    raise ValueError(f"Invalid source dataset: {SOURCE_DATASET}")

# Output directory for tokenized and batched dataset
if MAX_DATA_SIZE is not None:
    if MAX_DATA_SIZE // 1000 > 0:
        OUTPUT_DIR = f'/usr/xtmp/mth45/Documents/programmable_biology_group/cope/data/gfp/gfp_dataset_esm2_tokenized_bs{BATCH_SIZE}_leq{MAX_LENGTH}_n{MAX_DATA_SIZE//1000}k'
    else:
        OUTPUT_DIR = f'/usr/xtmp/mth45/Documents/programmable_biology_group/cope/data/gfp/gfp_dataset_esm2_tokenized_bs{BATCH_SIZE}_leq{MAX_LENGTH}_n{MAX_DATA_SIZE}'
else:
    OUTPUT_DIR = f'/usr/xtmp/mth45/Documents/programmable_biology_group/cope/data/gfp/gfp_dataset_esm2_tokenized_bs{BATCH_SIZE}_leq{MAX_LENGTH}'
# ------------------------------------------------------------
# 1) Load sequences from pre-split FASTA files
# ------------------------------------------------------------
from Bio import SeqIO

def load_and_filter_sequences(fasta_path, split_name):
    """Load sequences from a FASTA file and filter them."""
    sequences = []
    print(f"Loading {split_name} sequences from {fasta_path}...")
    for record in SeqIO.parse(fasta_path, "fasta"):
        s = str(record.seq)
        # skip empty sequences
        if not s:
            continue
        sequences.append(s)
    
    # Remove duplicates within this split
    sequences = list(set(sequences))
    
    # Filter out any sequence that contains non-natural amino acids
    print(f"len({split_name}) before filtering: {len(sequences)}")
    sequences = [
        seq for seq in sequences
        if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq)
        and (MAX_LENGTH is None or len(seq) <= MAX_LENGTH)
        and (MIN_LENGTH is None or len(seq) >= MIN_LENGTH)
    ]
    print(f"len({split_name}) after filtering: {len(sequences)}")
    if sequences:
        print(f"Average length of {split_name} sequences: {sum(len(seq) for seq in sequences) / len(sequences):.1f}")
    
    return sequences

# Load sequences from each split
train_sequences = load_and_filter_sequences(TRAIN_FASTA, "train")
val_sequences = load_and_filter_sequences(VAL_FASTA, "val")
test_sequences = load_and_filter_sequences(TEST_FASTA, "test")

# Apply MAX_DATA_SIZE truncation if specified (applies to each split independently)
if MAX_DATA_SIZE is not None:
    print(f"Truncating each split to {MAX_DATA_SIZE} sequences")
    train_sequences = train_sequences[:MAX_DATA_SIZE]
    val_sequences = val_sequences[:MAX_DATA_SIZE]
    test_sequences = test_sequences[:MAX_DATA_SIZE]
    print(f"len(train_sequences) after truncation: {len(train_sequences)}")
    print(f"len(val_sequences) after truncation: {len(val_sequences)}")
    print(f"len(test_sequences) after truncation: {len(test_sequences)}")

# Apply NUM_REPEATS if specified
if NUM_REPEATS is not None:
    print(f"Repeating each split {NUM_REPEATS} times")
    train_sequences = train_sequences * NUM_REPEATS
    val_sequences = val_sequences * NUM_REPEATS
    test_sequences = test_sequences * NUM_REPEATS

print(f"Final len(train_sequences): {len(train_sequences)}")
print(f"Final len(val_sequences): {len(val_sequences)}")
print(f"Final len(test_sequences): {len(test_sequences)}")

# ---- Build vocab from tokenizer ----
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
print(f"Vocabulary Size: {tokenizer.vocab_size}")
# ------------------------------------------------------------
# 2) helper: tokenize SELFIES using the built tokenizer
# ------------------------------------------------------------
def tokenize_sequences(sequences):
    enc = {"input_ids": [], "attention_mask": []}
    for s in sequences:
        res = tokenizer(s, add_special_tokens=True)
        enc["input_ids"].append(res["input_ids"])
        enc["attention_mask"].append(res["attention_mask"])
    return enc


# ------------------------------------------------------------
# 3) build batched dataset (sorted by length, fixed B=64, pad in-batch)
# ------------------------------------------------------------
from tqdm import tqdm

def build_batched_dataset(sequences, batch_size=64, pad_id=0):
    toks = tokenize_sequences(sequences)
    input_ids_list = toks["input_ids"]
    attn_mask_list = toks["attention_mask"]

    # collect items with length
    items = []
    for ids, mask in tqdm(zip(input_ids_list, attn_mask_list), total=len(input_ids_list), desc="Collecting items with length"):
        items.append(
            {
                "input_ids": ids,
                "attention_mask": mask,
                "length": len(ids),
            }
        )

    # sort globally by length (ascending)
    items.sort(key=lambda x: x["length"])

    batched_input_ids = []
    batched_attention_masks = []
    batched_seq_lengths = []
    batched_batch_sizes = []

    n = len(items)
    iter_total = (n + batch_size - 1) // batch_size  # number of batches
    for start in tqdm(range(0, n, batch_size), total=iter_total, desc="Batching and padding"):
        batch_items = items[start:start + batch_size]

        # max length in this batch
        max_len_in_batch = max(it["length"] for it in batch_items)

        cur_ids = []
        cur_masks = []

        for it in batch_items:
            ids = it["input_ids"]
            mask = it["attention_mask"]

            # pad to max_len_in_batch
            pad_len = max_len_in_batch - len(ids)
            if pad_len > 0:
                ids = ids + [pad_id] * pad_len
                mask = mask + [0] * pad_len

            cur_ids.append(ids)    # now len == max_len_in_batch
            cur_masks.append(mask) # now len == max_len_in_batch

        batched_input_ids.append(cur_ids)               # (B, max_len_in_batch)
        batched_attention_masks.append(cur_masks)       # (B, max_len_in_batch)
        batched_seq_lengths.append(max_len_in_batch)    # store the padded length
        batched_batch_sizes.append(len(batch_items))    # <= batch_size

    ds = Dataset.from_dict(
        {
            "input_ids": batched_input_ids,          # list of (B, Lpadded)
            "attention_mask": batched_attention_masks,
            "seq_length": batched_seq_lengths,       # Lpadded
            "batch_size": batched_batch_sizes,
        }
    )
    return ds


# ------------------------------------------------------------
# 4) build batched datasets for each split
# ------------------------------------------------------------
train_ds = build_batched_dataset(train_sequences, batch_size=BATCH_SIZE, pad_id=PAD_ID)
val_ds   = build_batched_dataset(val_sequences,   batch_size=BATCH_SIZE, pad_id=PAD_ID)
test_ds  = build_batched_dataset(test_sequences,  batch_size=BATCH_SIZE, pad_id=PAD_ID)

for ds in [train_ds, val_ds, test_ds]:
    num_seqs = 0
    for item in ds:
        num_seqs += len(item["input_ids"])
    print(f"Number of sequences in {ds}: {num_seqs}")

dsdict = DatasetDict(
    {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
dsdict.save_to_disk(OUTPUT_DIR)
print(f"saved to {OUTPUT_DIR}")
