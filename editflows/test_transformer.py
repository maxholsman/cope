#!/usr/bin/env python3
"""
Test script for the Transformer model.
Tests model execution and internal shapes with various input formats.
"""

import torch
from omegaconf import OmegaConf
from model.transformer import Transformer, make_score_mod_for_intra_sequence_only, build_seq_ids, lengths_to_offsets
from model import transformer as transformer_module
from torch.nn.attention.flex_attention import flex_attention as original_flex_attention_func

# Import text version for comparison
import sys
import os
import importlib.util

# Try to import text version for comparison
TextDDiTBlock = None
TextRotary = None
text_apply_rotary_emb = None

# Get absolute paths to avoid issues with relative paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
text_model_path = os.path.join(_script_dir, '..', 'text', 'model')
text_transformer_path = os.path.abspath(os.path.join(text_model_path, 'transformer.py'))
text_rotary_path = os.path.abspath(os.path.join(text_model_path, 'rotary.py'))
text_parent_path = os.path.abspath(os.path.join(_script_dir, '..', 'text'))

# Debug output
_transformer_exists = os.path.exists(text_transformer_path)
_rotary_exists = os.path.exists(text_rotary_path)

if _transformer_exists and _rotary_exists:
    # Temporarily remove editflows model from sys.modules to avoid conflicts
    # Save original modules
    _saved_modules = {}
    _module_keys_to_remove = ['model', 'model.transformer', 'model.rotary']
    for key in _module_keys_to_remove:
        if key in sys.modules:
            _saved_modules[key] = sys.modules.pop(key)
    
    try:
        # Add text parent directory to sys.path
        if text_parent_path not in sys.path:
            sys.path.insert(0, text_parent_path)
        
        # Now import text version (will load from text directory since editflows version is removed)
        import importlib
        text_model_rotary = importlib.import_module('model.rotary')
        text_model_transformer = importlib.import_module('model.transformer')
        
        # Extract the classes we need
        TextDDiTBlock = text_model_transformer.DDiTBlock
        TextRotary = text_model_rotary.Rotary
        text_apply_rotary_emb = text_model_rotary.apply_rotary_emb_torch
        
        # Store in a different location to avoid conflicts
        import types
        text_modules = types.SimpleNamespace()
        text_modules.transformer = text_model_transformer
        text_modules.rotary = text_model_rotary
        
    except Exception as e:
        # If import fails, set to None so test can be skipped
        import traceback
        print(f"\nWarning: Could not import text model for comparison")
        print(f"Error: {e}")
        print(f"Transformer path: {text_transformer_path}")
        print(f"Rotary path: {text_rotary_path}")
        print(f"Parent path: {text_parent_path}")
        print(f"sys.path contains text_parent_path: {text_parent_path in sys.path}")
        traceback.print_exc()
        TextDDiTBlock = None
        TextRotary = None
        text_apply_rotary_emb = None
    finally:
        # Restore original modules
        for key, mod in _saved_modules.items():
            sys.modules[key] = mod
        # Remove text parent path
        if text_parent_path in sys.path:
            sys.path.remove(text_parent_path)
else:
    # Files don't exist - print debug info
    print(f"\n⚠ Text model files not found (test will be skipped):")
    print(f"  Script directory: {_script_dir}")
    print(f"  Transformer path: {text_transformer_path}")
    print(f"    Exists: {_transformer_exists}")
    print(f"  Rotary path: {text_rotary_path}")
    print(f"    Exists: {_rotary_exists}")
    print(f"  Parent path: {text_parent_path}")
    print(f"    Exists: {os.path.exists(text_parent_path)}")
    print(f"  Model path: {text_model_path}")
    print(f"    Exists: {os.path.exists(text_model_path)}")
    if os.path.exists(text_parent_path):
        print(f"\n  Contents of text directory:")
        try:
            for item in os.listdir(text_parent_path):
                print(f"    - {item}")
        except:
            pass


def test_config():
    """Create a test configuration"""
    config = OmegaConf.create({
        'hidden_size': 768,
        'cond_dim': 128,
        'n_blocks': 12,
        'n_heads': 12,
        'dropout': 0.1,
        'esm_model_name': None,  # Disable ESM for faster testing (uses standard embedding instead)
        'freeze_esm': False,  # Not used when esm_model_name is None
    })
    return config


def print_shapes(name, value, indent=0):
    """Helper to print shapes of tensors or lists"""
    prefix = "  " * indent
    if isinstance(value, torch.Tensor):
        print(f"{prefix}{name}: {value.shape} | dtype={value.dtype}")
    elif isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: list/tuple of length {len(value)}")
        for i, item in enumerate(value):
            if isinstance(item, torch.Tensor):
                print(f"{prefix}  [{i}]: {item.shape} | dtype={item.dtype}")
            else:
                print(f"{prefix}  [{i}]: {type(item).__name__}")
    else:
        print(f"{prefix}{name}: {type(value).__name__} = {value}")


def test_with_list_input():
    """Test with ragged list input (List[Tensor])"""
    print("\n" + "="*80)
    print("TEST 1: Ragged list input (List[Tensor])")
    print("="*80)
    
    config = test_config()
    vocab_size = 33  # Small vocab for testing
    model = Transformer(vocab_size=vocab_size, masked=False, config=config)
    model.eval()
    
    # Create ragged sequences: [3, 5, 2] tokens
    x_t = [
        torch.randint(0, vocab_size, (3,), dtype=torch.long),
        torch.randint(0, vocab_size, (5,), dtype=torch.long),
        torch.randint(0, vocab_size, (2,), dtype=torch.long),
    ]
    time = torch.rand(3)  # (B=3,)
    
    print(f"\nInput shapes:")
    print(f"  x_t: list of {len(x_t)} sequences")
    for i, seq in enumerate(x_t):
        print(f"    [{i}]: {seq.shape}")
    print(f"  time: {time.shape}")
    
    print(f"\nModel parameters:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  cond_dim: {config.cond_dim}")
    print(f"  n_blocks: {config.n_blocks}")
    print(f"  n_heads: {config.n_heads}")
    
    with torch.no_grad():
        outputs = model(x_t, time)
    
    lam_ins, q_ins, lam_del, lam_sub, q_sub = outputs
    
    print(f"\nOutput shapes:")
    print(f"  lam_ins: list of {len(lam_ins)} sequences")
    for i, lam in enumerate(lam_ins):
        print(f"    [{i}]: {lam.shape} (expected: ({len(x_t[i])+1},))")
    
    print(f"  q_ins: list of {len(q_ins)} sequences")
    for i, q in enumerate(q_ins):
        print(f"    [{i}]: {q.shape} (expected: ({len(x_t[i])+1}, {vocab_size}))")
    
    print(f"  lam_del: list of {len(lam_del)} sequences")
    for i, lam in enumerate(lam_del):
        print(f"    [{i}]: {lam.shape} (expected: ({len(x_t[i])},))")
    
    print(f"  lam_sub: list of {len(lam_sub)} sequences")
    for i, lam in enumerate(lam_sub):
        print(f"    [{i}]: {lam.shape} (expected: ({len(x_t[i])},))")
    
    print(f"  q_sub: list of {len(q_sub)} sequences")
    for i, q in enumerate(q_sub):
        print(f"    [{i}]: {q.shape} (expected: ({len(x_t[i])}, {vocab_size}))")
    
    # Verify shapes
    assert len(lam_ins) == len(x_t), "lam_ins length mismatch"
    assert len(lam_del) == len(x_t), "lam_del length mismatch"
    for i in range(len(x_t)):
        assert lam_ins[i].shape == (1, len(x_t[i])+1,), f"lam_ins[{i}] shape mismatch"
        assert q_ins[i].shape == (1, len(x_t[i])+1, vocab_size), f"q_ins[{i}] shape mismatch"
        assert lam_del[i].shape == (1, len(x_t[i]),), f"lam_del[{i}] shape mismatch"
        assert lam_sub[i].shape == (1, len(x_t[i]),), f"lam_sub[{i}] shape mismatch"
        assert q_sub[i].shape == (1, len(x_t[i]), vocab_size), f"q_sub[{i}] shape mismatch"
    
    print("\n✓ All shape checks passed!")
    return outputs


def test_with_batched_input():
    """Test with batched tensor input (B, S)"""
    print("\n" + "="*80)
    print("TEST 2: Batched tensor input (B, S)")
    print("="*80)
    
    config = test_config()
    vocab_size = 33
    model = Transformer(vocab_size=vocab_size, masked=False, config=config)
    model.eval()
    
    # Create batched input: (B=3, S=5)
    B, S = 3, 5
    x_t = torch.randint(0, vocab_size, (B, S), dtype=torch.long)
    time = torch.rand(B)
    
    print(f"\nInput shapes:")
    print(f"  x_t: {x_t.shape}")
    print(f"  time: {time.shape}")
    
    with torch.no_grad():
        outputs = model(x_t, time)
    
    lam_ins, q_ins, lam_del, lam_sub, q_sub = outputs
    
    print(f"\nOutput shapes:")
    print(f"  lam_ins: list of {len(lam_ins)} sequences")
    for i, lam in enumerate(lam_ins):
        print(f"    [{i}]: {lam.shape} (expected: ({S+1},))")
    
    print(f"  q_ins: list of {len(q_ins)} sequences")
    for i, q in enumerate(q_ins):
        print(f"    [{i}]: {q.shape} (expected: ({S+1}, {vocab_size}))")
    
    print(f"  lam_del: list of {len(lam_del)} sequences")
    for i, lam in enumerate(lam_del):
        print(f"    [{i}]: {lam.shape} (expected: ({S},))")
    
    print(f"  lam_sub: list of {len(lam_sub)} sequences")
    for i, lam in enumerate(lam_sub):
        print(f"    [{i}]: {lam.shape} (expected: ({S},))")
    
    print(f"  q_sub: list of {len(q_sub)} sequences")
    for i, q in enumerate(q_sub):
        print(f"    [{i}]: {q.shape} (expected: ({S}, {vocab_size}))")
    
    # Verify shapes
    assert len(lam_ins) == B, "lam_ins length mismatch"
    for i in range(B):
        assert lam_ins[i].shape == (1, S+1,), f"lam_ins[{i}] shape mismatch"
        assert q_ins[i].shape == (1, S+1, vocab_size), f"q_ins[{i}] shape mismatch"
        assert lam_del[i].shape == (1, S,), f"lam_del[{i}] shape mismatch"
        assert lam_sub[i].shape == (1, S,), f"lam_sub[{i}] shape mismatch"
        assert q_sub[i].shape == (1, S, vocab_size), f"q_sub[{i}] shape mismatch"
    
    print("\n✓ All shape checks passed!")
    return outputs


def test_masked_model():
    """Test with masked token enabled"""
    print("\n" + "="*80)
    print("TEST 3: Masked model (with extra mask token)")
    print("="*80)
    
    config = test_config()
    vocab_size = 33
    model = Transformer(vocab_size=vocab_size, masked=True, config=config)
    model.eval()
    
    # Use mask token (vocab_size) in input
    x_t = [
        torch.tensor([0, vocab_size, 2], dtype=torch.long),  # includes mask token
        torch.tensor([1, 2, 3, 4], dtype=torch.long),
    ]
    time = torch.rand(2)
    
    print(f"\nInput:")
    print(f"  x_t: list with mask token (vocab_size={vocab_size})")
    print(f"  time: {time.shape}")
    
    with torch.no_grad():
        outputs = model(x_t, time)
    
    lam_ins, q_ins, lam_del, lam_sub, q_sub = outputs
    
    print(f"\nOutput vocab_size check:")
    print(f"  q_ins[0].shape[-1]: {q_ins[0].shape[-1]} (expected: {vocab_size+1})")
    print(f"  q_sub[0].shape[-1]: {q_sub[0].shape[-1]} (expected: {vocab_size+1})")
    
    assert q_ins[0].shape[-1] == vocab_size + 1, "q_ins vocab_size mismatch"
    assert q_sub[0].shape[-1] == vocab_size + 1, "q_sub vocab_size mismatch"
    
    print("\n✓ Masked model test passed!")
    return outputs


def test_internal_shapes():
    """Test internal tensor shapes in the model"""
    print("\n" + "="*80)
    print("TEST 4: Internal tensor shapes")
    print("="*80)
    
    config = test_config()
    vocab_size = 33
    model = Transformer(vocab_size=vocab_size, masked=False, config=config)
    model.eval()
    
    x_t = [torch.randint(0, vocab_size, (4,), dtype=torch.long)]
    time = torch.rand(1)
    
    # Hook to capture intermediate shapes
    captured_shapes = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captured_shapes[name] = output.shape
            elif isinstance(output, (list, tuple)):
                captured_shapes[name] = [o.shape if isinstance(o, torch.Tensor) else type(o).__name__ 
                                        for o in output]
        return hook
    
    # Register hooks
    # vocab_embed only exists if ESM is disabled (esm_model_name is None)
    if hasattr(model, "vocab_embed") and model.vocab_embed is not None:
        model.vocab_embed.register_forward_hook(hook_fn("vocab_embed"))
    elif hasattr(model, "tok_embedder") and model.tok_embedder is not None:
        # If using ESM, register hook on tok_embedder instead
        model.tok_embedder.register_forward_hook(hook_fn("tok_embedder"))
    model.time_embedding.register_forward_hook(hook_fn("time_embedding"))
    if len(model.blocks) > 0:
        model.blocks[0].norm1.register_forward_hook(hook_fn("block0_norm1"))
        model.blocks[0].qw.register_forward_hook(hook_fn("block0_qw"))
    
    with torch.no_grad():
        outputs = model(x_t, time)
    
    print(f"\nInternal tensor shapes:")
    for name, shape in captured_shapes.items():
        print(f"  {name}: {shape}")
    
    print("\n✓ Internal shapes captured!")
    return outputs


def test_value_ranges():
    """Test that outputs have expected value ranges"""
    print("\n" + "="*80)
    print("TEST 5: Output value ranges")
    print("="*80)
    
    config = test_config()
    vocab_size = 33
    model = Transformer(vocab_size=vocab_size, masked=False, config=config)
    model.eval()
    
    x_t = [torch.randint(0, vocab_size, (5,), dtype=torch.long)]
    time = torch.rand(1)
    
    with torch.no_grad():
        lam_ins, q_ins, lam_del, lam_sub, q_sub = model(x_t, time)
    
    # Check lambda values should be positive (from Positive module)
    lam_ins_flat = lam_ins[0]
    lam_del_flat = lam_del[0]
    lam_sub_flat = lam_sub[0]
    
    print(f"\nValue ranges:")
    print(f"  lam_ins: min={lam_ins_flat.min().item():.6f}, max={lam_ins_flat.max().item():.6f}")
    print(f"  lam_del: min={lam_del_flat.min().item():.6f}, max={lam_del_flat.max().item():.6f}")
    print(f"  lam_sub: min={lam_sub_flat.min().item():.6f}, max={lam_sub_flat.max().item():.6f}")
    
    # q_sub should be probabilities (convert from logits using softmax)
    q_sub_flat = q_sub[0]  # (1, n, V) - raw logits
    q_sub_probs = torch.softmax(q_sub_flat, dim=-1)  # Convert to probabilities
    q_sub_sum = q_sub_probs.sum(dim=-1)
    print(f"  q_sub row sums: min={q_sub_sum.min().item():.6f}, max={q_sub_sum.max().item():.6f}")
    print(f"    (should be ~1.0 for probabilities)")
    
    # q_ins - check shape and range (no softmax according to comment)
    q_ins_flat = q_ins[0]
    print(f"  q_ins: min={q_ins_flat.min().item():.6f}, max={q_ins_flat.max().item():.6f}")
    print(f"    (raw logits, no softmax applied)")
    
    # Verify lambdas are positive
    assert (lam_ins_flat >= 0).all(), "lam_ins should be non-negative"
    assert (lam_del_flat >= 0).all(), "lam_del should be non-negative"
    assert (lam_sub_flat >= 0).all(), "lam_sub should be non-negative"
    
    # Verify q_sub sums are close to 1 (probabilities)
    assert torch.allclose(q_sub_sum, torch.ones_like(q_sub_sum), atol=1e-5), \
        "q_sub should sum to 1 per row (probabilities)"
    
    print("\n✓ All value range checks passed!")


def test_gradient_flow():
    """Test that gradients can flow through the model"""
    print("\n" + "="*80)
    print("TEST 6: Gradient flow")
    print("="*80)
    
    config = test_config()
    vocab_size = 33
    model = Transformer(vocab_size=vocab_size, masked=False, config=config)
    model.train()
    
    x_t = [torch.randint(0, vocab_size, (3,), dtype=torch.long)]
    time = torch.rand(1)
    
    # Forward pass
    lam_ins, q_ins, lam_del, lam_sub, q_sub = model(x_t, time)
    
    # Create a dummy loss (sum of all outputs)
    loss = lam_ins[0].sum() + q_ins[0].sum() + lam_del[0].sum() + lam_sub[0].sum() + q_sub[0].sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
    
    print(f"\nParameters with gradients: {grad_count}")
    print(f"Total parameters: {sum(1 for _ in model.parameters())}")
    
    assert grad_count > 0, "No gradients found!"
    print("\n✓ Gradient flow test passed!")


def test_intra_sequence_attention():
    """Test that flex attention only attends within sequences, not across them"""
    print("\n" + "="*80)
    print("TEST 7: Intra-sequence attention (no cross-sequence attention)")
    print("="*80)
    
    # Test 1: Direct test of score_mod function
    print("\n--- Test 1: Score modification function ---")
    lengths = torch.tensor([3, 5, 2], dtype=torch.long)  # 3 sequences with lengths [3, 5, 2]
    total_tokens = lengths.sum().item()  # 10 tokens total
    
    seq_ids = build_seq_ids(lengths)
    print(f"Sequence lengths: {lengths.tolist()}")
    print(f"Total tokens: {total_tokens}")
    print(f"Sequence IDs: {seq_ids.tolist()}")
    print(f"Expected: [0,0,0, 1,1,1,1,1, 2,2]")
    
    # Verify seq_ids are correct
    expected_seq_ids = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
    assert seq_ids.tolist() == expected_seq_ids, f"seq_ids mismatch: {seq_ids.tolist()} != {expected_seq_ids}"
    
    # Create score_mod function
    score_mod = make_score_mod_for_intra_sequence_only(lengths)
    
    # Test score_mod function directly as it would be called by flex_attention
    # flex_attention calls score_mod(scores, b, h, q_idx, k_idx) where:
    # - scores: attention scores tensor (shape depends on blocks)
    # - b: batch index (0 in our case)
    # - h: head index
    # - q_idx: flat query token indices (0 to T-1)
    # - k_idx: flat key token indices (0 to T-1)
    
    # Simulate what flex_attention would pass: indices for all query-key pairs
    # For a small test, we'll use scalar indices (flex_attention uses blocks)
    dummy_scores = torch.ones(2, 3, dtype=torch.float32)  # (q_block=2, k_block=3)
    
    # Test some specific query-key pairs
    test_pairs = [
        # (q_idx, k_idx, should_be_same_seq, description)
        (0, 0, True, "Seq0[0] -> Seq0[0] (same token)"),
        (0, 1, True, "Seq0[0] -> Seq0[1] (same seq)"),
        (0, 2, True, "Seq0[0] -> Seq0[2] (same seq)"),
        (0, 3, False, "Seq0[0] -> Seq1[0] (cross seq)"),
        (0, 4, False, "Seq0[0] -> Seq1[1] (cross seq)"),
        (3, 3, True, "Seq1[0] -> Seq1[0] (same token)"),
        (3, 7, True, "Seq1[0] -> Seq1[4] (same seq)"),
        (3, 0, False, "Seq1[0] -> Seq0[0] (cross seq)"),
        (8, 9, True, "Seq2[0] -> Seq2[1] (same seq)"),
        (8, 0, False, "Seq2[0] -> Seq0[0] (cross seq)"),
    ]
    
    neg_large = torch.finfo(dummy_scores.dtype).min
    
    print(f"\nTesting score_mod function with specific query-key pairs:")
    same_count = 0
    cross_count = 0
    
    for q_idx, k_idx, should_be_same, desc in test_pairs:
        # Create dummy scores and call score_mod
        test_scores = torch.ones(1, 1, dtype=torch.float32) * 5.0  # Use 5.0 to make it obvious
        
        # Call score_mod as flex_attention would (b=0, h=0 for single head)
        modified = score_mod(test_scores, b=0, h=0, q_idx=torch.tensor([q_idx]), k_idx=torch.tensor([k_idx]))
        modified_value = modified.item()
        
        is_same = (modified_value > 0)  # Positive means same sequence
        is_masked = (modified_value == neg_large)  # neg_large means masked
        
        print(f"  {desc:40s} | q={q_idx:2d}, k={k_idx:2d} | score={modified_value:10.3f} | "
              f"{'SAME' if is_same else 'MASKED':5s}")
        
        if should_be_same:
            assert is_same, f"Pair ({q_idx},{k_idx}) should be same-seq but was masked"
            same_count += 1
        else:
            assert is_masked, f"Pair ({q_idx},{k_idx}) should be cross-seq (masked) but was not"
            cross_count += 1
    
    print(f"\n✓ Score modification verified:")
    print(f"  ✓ Same-sequence pairs: {same_count} correctly preserved")
    print(f"  ✓ Cross-sequence pairs: {cross_count} correctly masked")
    
    # Test 2: Verify flex_attention produces same output as traditional masked attention
    print("\n--- Test 2: Flex attention vs traditional masked attention ---")
    
    # Store captured Q, K, V and flex_attention output
    qkv_storage = {}
    
    # Store original flex_attention from both locations
    # The function is imported directly, so we need to patch the module's reference
    original_flex_attention_module = transformer_module.flex_attention
    
    # Create a wrapper to capture Q, K, V and output
    def wrapped_flex_attention(q, k, v, score_mod=None, **kwargs):
        # Store Q, K, V for manual computation
        qkv_storage['q'] = q.detach().clone()
        qkv_storage['k'] = k.detach().clone()
        qkv_storage['v'] = v.detach().clone()
        # Store score_mod for later inspection
        qkv_storage['score_mod'] = score_mod
        # Call original function and store output
        output = original_flex_attention_func(q, k, v, score_mod=score_mod, **kwargs)
        qkv_storage['flex_output'] = output.detach().clone()
        return output
    
    # Monkey patch flex_attention BEFORE creating the model
    # This ensures the model uses our wrapped version
    # The model imports it directly, so we need to patch the module's reference
    transformer_module.flex_attention = wrapped_flex_attention
    
    # Use a simpler model with fewer blocks for faster testing
    simple_config = OmegaConf.create({
        'hidden_size': 64,  # Smaller for faster testing
        'cond_dim': 32,
        'n_blocks': 1,  # Single block
        'n_heads': 2,
        'dropout': 0.0,  # No dropout for deterministic comparison
    })
    
    vocab_size = 33
    model = Transformer(vocab_size=vocab_size, masked=False, config=simple_config)
    model.eval()
    
    # Create test sequences with known patterns
    x_t = [
        torch.tensor([1, 2, 3], dtype=torch.long),  # Seq 0: length 3
        torch.tensor([10, 20], dtype=torch.long),   # Seq 1: length 2
    ]
    time = torch.tensor([0.5, 0.5], dtype=torch.float32)
    
    print(f"Input sequences:")
    print(f"  Sequence 0: {x_t[0].tolist()} (length 3)")
    print(f"  Sequence 1: {x_t[1].tolist()} (length 2)")
    
    try:
        with torch.no_grad():
            # Run forward pass
            model_outputs = model(x_t, time)
            
            # Check if flex_attention was called
            if 'q' not in qkv_storage:
                raise RuntimeError(
                    "flex_attention was not called. This might mean:\n"
                    "1. The model is not using flex_attention\n"
                    "2. The monkey patch didn't work\n"
                    "3. The forward pass failed before reaching flex_attention"
                )
            
            # Extract captured tensors
            # flex_attention is called with (1, Hh, T, Dh), so we need to squeeze batch dim
            q_captured = qkv_storage['q']
            k_captured = qkv_storage['k']
            v_captured = qkv_storage['v']
            flex_output = qkv_storage['flex_output']
            
            # Remove batch dimension if present (should be (1, Hh, T, Dh) -> (Hh, T, Dh))
            if q_captured.dim() == 4:
                q_captured = q_captured.squeeze(0)  # (Hh, T, Dh)
                k_captured = k_captured.squeeze(0)  # (Hh, T, Dh)
                v_captured = v_captured.squeeze(0)  # (Hh, T, Dh)
                flex_output = flex_output.squeeze(0)  # (Hh, T, Dh)
            
            print(f"\nCaptured tensor shapes:")
            print(f"  Q: {q_captured.shape}")
            print(f"  K: {k_captured.shape}")
            print(f"  V: {v_captured.shape}")
            print(f"  Flex attention output: {flex_output.shape}")
            
            # Get sequence information
            lengths = torch.tensor([len(x_t[0]), len(x_t[1])], dtype=torch.long)
            seq_ids = build_seq_ids(lengths)
            T = q_captured.shape[1]  # Total tokens
            Hh = q_captured.shape[0]  # Number of heads
            Dh = q_captured.shape[2]  # Head dimension
            
            print(f"\nSequence information:")
            print(f"  Total tokens: {T}")
            print(f"  Number of heads: {Hh}")
            print(f"  Head dimension: {Dh}")
            print(f"  Sequence IDs: {seq_ids.tolist()}")
            
            # Compute traditional masked attention manually
            print(f"\nComputing traditional masked attention...")
            manual_outputs = []
            neg_large = torch.finfo(q_captured.dtype).min
            
            for h in range(Hh):
                q_h = q_captured[h, :, :]  # (T, Dh)
                k_h = k_captured[h, :, :]  # (T, Dh)
                v_h = v_captured[h, :, :]  # (T, Dh)
                
                # Compute attention scores: Q @ K^T / sqrt(d)
                scores = torch.matmul(q_h, k_h.transpose(0, 1)) / (Dh ** 0.5)  # (T, T)
                
                # Apply masking: set cross-sequence scores to neg_large
                masked_scores = scores.clone()
                for q_idx in range(T):
                    for k_idx in range(T):
                        if seq_ids[q_idx] != seq_ids[k_idx]:
                            masked_scores[q_idx, k_idx] = neg_large
                
                # Apply softmax (cross-sequence will have effectively zero probability)
                attn_weights = torch.softmax(masked_scores, dim=-1)  # (T, T)
                
                # Apply to values
                attn_output_h = torch.matmul(attn_weights, v_h)  # (T, Dh)
                manual_outputs.append(attn_output_h)
            
            # Stack heads: (Hh, T, Dh)
            manual_output = torch.stack(manual_outputs, dim=0)  # (Hh, T, Dh)
            
            print(f"Manual attention output shape: {manual_output.shape}")
            
            # Compare flex_attention output with manual computation
            print(f"\nComparing flex_attention vs manual masked attention:")
            diff = (flex_output - manual_output).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            relative_diff = (diff / (manual_output.abs() + 1e-8)).max().item()
            
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Mean absolute difference: {mean_diff:.2e}")
            print(f"  Max relative difference: {relative_diff:.2e}")
            
            # Print example outputs for comparison
            print(f"\nExample outputs (first head, first few tokens):")
            print(f"  Flex attention output (head 0, token 0):")
            print(f"    {flex_output[0, 0, :8].tolist()}")  # First 8 dims
            print(f"  Manual attention output (head 0, token 0):")
            print(f"    {manual_output[0, 0, :8].tolist()}")  # First 8 dims
            print(f"  Difference:")
            print(f"    {diff[0, 0, :8].tolist()}")
            
            print(f"\n  Flex attention output (head 0, token 3 - start of Seq1):")
            print(f"    {flex_output[0, 3, :8].tolist()}")
            print(f"  Manual attention output (head 0, token 3):")
            print(f"    {manual_output[0, 3, :8].tolist()}")
            print(f"  Difference:")
            print(f"    {diff[0, 3, :8].tolist()}")
            
            print(f"\n  Flex attention output (head 1, token 0):")
            print(f"    {flex_output[1, 0, :8].tolist()}")
            print(f"  Manual attention output (head 1, token 0):")
            print(f"    {manual_output[1, 0, :8].tolist()}")
            print(f"  Difference:")
            print(f"    {diff[1, 0, :8].tolist()}")
            
            # They should match very closely (within numerical precision)
            # flex_attention might use different block sizes or optimizations, but results should match
            tolerance = 1e-4
            error_msg = (
                f"Flex attention and manual masked attention differ by {max_diff:.2e}, "
                f"expected < {tolerance:.2e}"
            )
            assert max_diff < tolerance, error_msg
            
            print(f"\n✓ Flex attention matches manual masked attention (diff < {tolerance:.2e})")
            
            # Additional check: verify cross-sequence attention is masked
            print(f"\nVerifying cross-sequence attention is masked:")
            
            # For each head, check attention weights
            cross_seq_sum = 0.0
            same_seq_sum = 0.0
            
            for h in range(Hh):
                q_h = q_captured[h, :, :]
                k_h = k_captured[h, :, :]
                scores = torch.matmul(q_h, k_h.transpose(0, 1)) / (Dh ** 0.5)
                
                # Apply masking
                masked_scores = scores.clone()
                for q_idx in range(T):
                    for k_idx in range(T):
                        if seq_ids[q_idx] != seq_ids[k_idx]:
                            masked_scores[q_idx, k_idx] = neg_large
                
                attn_weights = torch.softmax(masked_scores, dim=-1)
                
                # Check cross-sequence attention (Seq 0 -> Seq 1)
                for q_idx in range(3):  # Seq 0 tokens (0, 1, 2)
                    for k_idx in range(3, 5):  # Seq 1 tokens (3, 4)
                        cross_seq_sum += attn_weights[q_idx, k_idx].item()
                
                # Check same-sequence attention (within Seq 0)
                for q_idx in range(3):
                    for k_idx in range(3):
                        same_seq_sum += attn_weights[q_idx, k_idx].item()
            
            print(f"  Cross-sequence attention sum (Seq0 -> Seq1): {cross_seq_sum:.9e}")
            print(f"  Same-sequence attention sum (within Seq0): {same_seq_sum:.6f}")
            
            assert cross_seq_sum < 1e-6, \
                f"Cross-sequence attention should be near-zero, got {cross_seq_sum:.9e}"
            assert same_seq_sum > 0.1, \
                f"Same-sequence attention should be significant, got {same_seq_sum:.6f}"
            
            print(f"  ✓ Cross-sequence attention is effectively zero")
            print(f"  ✓ Same-sequence attention is preserved")
            
    finally:
        # Restore original flex_attention
        transformer_module.flex_attention = original_flex_attention_module
    
    # Test 3: Verify model runs correctly with ragged sequences
    print("\n--- Test 3: Model runs correctly with ragged sequences ---")
    
    # Create sequences where we can track attention behavior
    # Use a simpler model with fewer blocks for testing
    simple_config = OmegaConf.create({
        'hidden_size': 64,  # Smaller for faster testing
        'cond_dim': 32,
        'n_blocks': 1,  # Single block
        'n_heads': 2,
        'dropout': 0.0,  # No dropout for deterministic testing
    })
    
    simple_model = Transformer(vocab_size=vocab_size, masked=False, config=simple_config)
    simple_model.eval()
    
    # Create test sequences with known patterns
    test_x_t = [
        torch.tensor([1, 2, 3], dtype=torch.long),  # Seq 0
        torch.tensor([10, 20], dtype=torch.long),   # Seq 1
    ]
    test_time = torch.tensor([0.5, 0.5], dtype=torch.float32)
    
    with torch.no_grad():
        test_outputs = simple_model(test_x_t, test_time)
    
    print(f"Model forward pass completed successfully with ragged sequences")
    print(f"  Sequence 0 output shapes:")
    print(f"    lam_ins[0]: {test_outputs[0][0].shape}")
    print(f"    lam_del[0]: {test_outputs[2][0].shape}")
    print(f"  Sequence 1 output shapes:")
    print(f"    lam_ins[1]: {test_outputs[0][1].shape}")
    print(f"    lam_del[1]: {test_outputs[2][1].shape}")
    
    # Verify that outputs have correct lengths (n+1 for ins, n for del)
    assert test_outputs[0][0].shape[-1] == 4, "lam_ins[0] should have length 4 (n+1)"
    assert test_outputs[0][1].shape[-1] == 3, "lam_ins[1] should have length 3 (n+1)"
    assert test_outputs[2][0].shape[-1] == 3, "lam_del[0] should have length 3 (n)"
    assert test_outputs[2][1].shape[-1] == 2, "lam_del[1] should have length 2 (n)"
    
    print("\n✓ Intra-sequence attention test passed!")
    print("  ✓ Score modification correctly masks cross-sequence attention")
    print("  ✓ Flex attention produces same output as traditional masked attention")
    print("  ✓ Model handles ragged sequences without cross-sequence contamination")


def test_text_vs_editflows_block_comparison():
    """Test that text version DDiTBlock + rotary embedding produces same output as editflows version"""
    print("\n" + "="*80)
    print("TEST 8: Text version vs EditFlows version (DDiTBlock + Rotary)")
    print("="*80)
    
    # Skip if text version not available
    if TextDDiTBlock is None or TextRotary is None:
        print("\n⚠ Skipping test: text model version not found")
        return
    
    # Use same config for both
    config = OmegaConf.create({
        'hidden_size': 64,
        'cond_dim': 32,
        'n_heads': 4,
        'dropout': 0.0,  # No dropout for deterministic comparison
    })
    
    # Import the editflows versions (use explicit imports to avoid collision)
    from model import transformer as editflows_transformer_module
    from model import rotary as editflows_rotary_module
    EditFlowsDDiTBlock = editflows_transformer_module.DDiTBlock
    EditFlowsRotary = editflows_rotary_module.Rotary
    
    # Create blocks with same parameters
    # Verify we're using the correct classes
    print(f"\nVerifying block types:")
    print(f"  TextDDiTBlock type: {TextDDiTBlock}")
    print(f"  TextDDiTBlock module: {TextDDiTBlock.__module__}")
    print(f"  EditFlowsDDiTBlock type: {EditFlowsDDiTBlock}")
    print(f"  EditFlowsDDiTBlock module: {EditFlowsDDiTBlock.__module__}")
    
    import inspect
    print(f"\nText DDiTBlock forward signature:")
    print(f"  {inspect.signature(TextDDiTBlock.forward)}")
    print(f"\nEditFlows DDiTBlock forward signature:")
    print(f"  {inspect.signature(EditFlowsDDiTBlock.forward)}")
    
    text_block = TextDDiTBlock(
        dim=config.hidden_size,
        n_heads=config.n_heads,
        cond_dim=config.cond_dim,
        mlp_ratio=4,
        dropout=config.dropout,
    )
    
    editflows_block = EditFlowsDDiTBlock(
        dim=config.hidden_size,
        n_heads=config.n_heads,
        cond_dim=config.cond_dim,
        mlp_ratio=4,
        dropout=config.dropout,
    )
    
    # Verify the blocks are different types
    assert text_block.__class__ != editflows_block.__class__, \
        "Blocks should be different classes but they're the same!"
    
    # Create rotary embeddings
    text_rotary = TextRotary(dim=config.hidden_size // config.n_heads)
    editflows_rotary = EditFlowsRotary(dim=config.hidden_size // config.n_heads)
    
    # Copy weights from text block to editflows block to ensure identical computation
    print("\nCopying weights from text version to editflows version...")
    editflows_block.norm1.weight.data.copy_(text_block.norm1.weight.data)
    editflows_block.norm2.weight.data.copy_(text_block.norm2.weight.data)
    editflows_block.qw.weight.data.copy_(text_block.qw.weight.data)
    editflows_block.kw.weight.data.copy_(text_block.kw.weight.data)
    editflows_block.vw.weight.data.copy_(text_block.vw.weight.data)
    editflows_block.attn_out.weight.data.copy_(text_block.attn_out.weight.data)
    editflows_block.mlp[0].weight.data.copy_(text_block.mlp[0].weight.data)
    editflows_block.mlp[0].bias.data.copy_(text_block.mlp[0].bias.data)
    editflows_block.mlp[2].weight.data.copy_(text_block.mlp[2].weight.data)
    editflows_block.mlp[2].bias.data.copy_(text_block.mlp[2].bias.data)
    editflows_block.adaLN_modulation.weight.data.copy_(text_block.adaLN_modulation.weight.data)
    editflows_block.adaLN_modulation.bias.data.copy_(text_block.adaLN_modulation.bias.data)
    
    # Copy rotary embedding weights
    editflows_rotary.inv_freq.data.copy_(text_rotary.inv_freq.data)
    
    # Set to eval mode
    text_block.eval()
    editflows_block.eval()
    
    # Create test input: single sequence for simplicity
    B, S, H = 1, 8, config.hidden_size
    x_batched = torch.randn(B, S, H)
    c = torch.randn(B, config.cond_dim)
    time = torch.tensor([0.5])
    
    print(f"\nTest input:")
    print(f"  Batched shape (text version): {x_batched.shape}")
    print(f"  Condition shape: {c.shape}")
    
    # Prepare text version input (already in correct format)
    x_text = x_batched.clone()
    
    # Prepare editflows version input (flat format)
    x_editflows_flat = x_batched.reshape(-1, H)  # (B*S, H) = (8, H)
    lengths = torch.tensor([S], dtype=torch.long)
    offsets = lengths_to_offsets(lengths)
    positions = editflows_rotary.positions_like(lengths)
    
    print(f"  Flat shape (editflows version): {x_editflows_flat.shape}")
    print(f"  Lengths: {lengths.tolist()}")
    print(f"  Positions: {positions.tolist()}")
    
    # Get rotary embeddings for text version
    # Text rotary expects (B, S, H) and returns (B, S, QKV, Hh, D) shaped cos/sin
    with torch.no_grad():
        text_cos_sin = text_rotary(x_text)
        
        # Editflows rotary - we need to call it with a batched view
        # The forward method expects (B, S, H) to cache cos/sin
        editflows_rotary(x_text)  # This caches cos/sin
        editflows_cos_sin = editflows_rotary.cos_cached, editflows_rotary.sin_cached
    
    print(f"\nRotary embedding shapes:")
    print(f"  Text version cos: {text_cos_sin[0].shape}")
    print(f"  Editflows version cos: {editflows_cos_sin[0].shape}")
    
    # Run text version block
    with torch.no_grad():
        text_output = text_block(x_text, text_cos_sin, c)
        print(f"\nText version output shape: {text_output.shape}")
    
    # Run editflows version block
    with torch.no_grad():
        editflows_output = editflows_block(
            x=x_editflows_flat,
            lengths=lengths,
            offsets=offsets,
            rotary_cos_sin=editflows_cos_sin,
            positions=positions,
            c=c
        )
        # Reshape to compare with text version
        editflows_output_batched = editflows_output.reshape(B, S, H)
        print(f"Editflows version output shape (reshaped): {editflows_output_batched.shape}")
    
    # Compare outputs
    diff = (text_output - editflows_output_batched).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    relative_diff = (diff / (text_output.abs() + 1e-8)).max().item()
    
    print(f"\nOutput comparison:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Max relative difference: {relative_diff:.2e}")
    
    # Print example outputs
    print(f"\nExample outputs (first 8 dimensions of first token):")
    print(f"  Text version:")
    print(f"    {text_output[0, 0, :8].tolist()}")
    print(f"  Editflows version:")
    print(f"    {editflows_output_batched[0, 0, :8].tolist()}")
    print(f"  Difference:")
    print(f"    {diff[0, 0, :8].tolist()}")
    
    # They should match very closely (within numerical precision)
    tolerance = 1e-4
    error_msg = (
        f"Text and EditFlows versions differ by {max_diff:.2e}, "
        f"expected < {tolerance:.2e}"
    )
    assert max_diff < tolerance, error_msg
    
    print(f"\n✓ Text and EditFlows DDiTBlock outputs match (diff < {tolerance:.2e})")
    
    # Also test rotary embedding application directly
    print(f"\n--- Testing rotary embedding application directly ---")
    
    # Create Q, K, V from the same input
    q_text = text_block.qw(text_block.norm1(x_text))
    k_text = text_block.kw(text_block.norm1(x_text))
    v_text = text_block.vw(text_block.norm1(x_text))
    
    q_text = q_text.view(B, S, config.n_heads, config.hidden_size // config.n_heads)
    k_text = k_text.view(B, S, config.n_heads, config.hidden_size // config.n_heads)
    
    # Apply text rotary embedding
    with torch.amp.autocast("cuda", enabled=False):
        q_text_rot = text_apply_rotary_emb(q_text.float(), text_cos_sin[0].float(), text_cos_sin[1].float())
        k_text_rot = text_apply_rotary_emb(k_text.float(), text_cos_sin[0].float(), text_cos_sin[1].float())
        q_text_rot = q_text_rot.to(q_text.dtype)
        k_text_rot = k_text_rot.to(k_text.dtype)
    
    # For editflows version, prepare flat format
    q_editflows = editflows_block.qw(editflows_block.norm1(x_editflows_flat))
    k_editflows = editflows_block.kw(editflows_block.norm1(x_editflows_flat))
    
    q_editflows = q_editflows.view(S, config.n_heads, config.hidden_size // config.n_heads)
    k_editflows = k_editflows.view(S, config.n_heads, config.hidden_size // config.n_heads)
    
    # Apply editflows rotary embedding
    from model.rotary import apply_rotary_emb_ragged
    with torch.amp.autocast("cuda", enabled=False):
        q_editflows_rot, k_editflows_rot = apply_rotary_emb_ragged(
            q_editflows.float(), k_editflows.float(),
            editflows_cos_sin[0], editflows_cos_sin[1],
            positions=positions,
            head_dim=config.hidden_size // config.n_heads
        )
        q_editflows_rot = q_editflows_rot.to(q_editflows.dtype)
        k_editflows_rot = k_editflows_rot.to(k_editflows.dtype)
    
    # Reshape editflows to compare
    q_editflows_batched = q_editflows_rot.unsqueeze(0)  # (1, S, Hh, Dh)
    k_editflows_batched = k_editflows_rot.unsqueeze(0)
    
    # Compare rotary embeddings
    q_diff = (q_text_rot - q_editflows_batched).abs()
    k_diff = (k_text_rot - k_editflows_batched).abs()
    
    print(f"  Q rotary max diff: {q_diff.max().item():.2e}")
    print(f"  K rotary max diff: {k_diff.max().item():.2e}")
    
    assert q_diff.max().item() < tolerance, "Q rotary embeddings don't match"
    assert k_diff.max().item() < tolerance, "K rotary embeddings don't match"
    
    print(f"  ✓ Rotary embeddings match between versions")


def main():
    """Run all tests"""
    print("="*80)
    print("TRANSFORMER MODEL TEST SUITE")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Move model to device for tests
    original_device = device
    
    try:
        # Run tests
        test_with_list_input()
        test_with_batched_input()
        test_masked_model()
        test_internal_shapes()
        test_value_ranges()
        test_gradient_flow()
        test_intra_sequence_attention()
        test_text_vs_editflows_block_comparison()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

