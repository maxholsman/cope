import os
import random
import pandas as pd
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
FASTA_PATH = '/usr/xtmp/mth45/Documents/programmable_biology_group/gated-edit-proposal/data/cas9/cas9_dataset_fasta.fa'
OUTPUT_DIR = '/usr/xtmp/mth45/Documents/programmable_biology_group/gated-edit-proposal/data/cas9/cas9_dataset_esm2_tokenized_under_500'  # where to save with save_to_disk

BATCH_SIZE = 1
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

PAD_ID = 0

# ------------------------------------------------------------
# 1) Load CSV and convert SMILES -> SELFIES
# ------------------------------------------------------------
from Bio import SeqIO

sequences = []
print("Loading sequences from FASTA and validating…")
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    s = str(record.seq)
    # skip empty sequences
    if not s:
        continue
    sequences.append(s)

# remove all duplicates 
sequences = list(set(sequences))

## filter out any sequence that contains non-natural amino acids
print(f"len(sequences) before filtering: {len(sequences)}")
sequences = [
    seq for seq in sequences
    if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq) and len(seq) <= 500
]
print(f"len(sequences) after filtering: {len(sequences)}")

# ---- Build vocab from all valid SELFIES ----
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
print(f"Vocabulary Size: {tokenizer.vocab_size}")

# shuffle before splitting
random.shuffle(sequences)
n = len(sequences)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_sequences = sequences[:n_train]
val_sequences   = sequences[n_train:n_train + n_val]
test_sequences  = sequences[n_train + n_val:]

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
