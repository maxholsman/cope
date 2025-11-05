#!/usr/bin/env python3
"""
Print token_id -> token mapping for ESM tokenizer
"""

from transformers import EsmTokenizer

def main():
    # Load ESM tokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"\nToken ID -> Token Mapping:")
    print("-" * 60)
    
    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    
    # Sort by token ID for easier reading
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    for token, token_id in sorted_vocab:
        print(f"ID {token_id:3d} -> {repr(token)}")
    
    print("-" * 60)
    print(f"\nTotal tokens: {len(sorted_vocab)}")
    
    # Also print special tokens if available
    print("\nSpecial tokens:")
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
        print(f"  PAD: {tokenizer.pad_token_id} -> {repr(tokenizer.pad_token)}")
    if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
        print(f"  BOS: {tokenizer.bos_token_id} -> {repr(tokenizer.bos_token)}")
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
        print(f"  EOS: {tokenizer.eos_token_id} -> {repr(tokenizer.eos_token)}")
    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token:
        print(f"  CLS: {tokenizer.cls_token_id} -> {repr(tokenizer.cls_token)}")
    if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
        print(f"  UNK: {tokenizer.unk_token_id} -> {repr(tokenizer.unk_token)}")
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
        print(f"  MASK: {tokenizer.mask_token_id} -> {repr(tokenizer.mask_token)}")

if __name__ == '__main__':
    main()





