from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

Tensor = torch.Tensor


def _optimal_align_1d(seq0, seq1, eps_id):
    """
    seq0: (L0,) 1D LongTensor (no pad)
    seq1: (L1,) 1D LongTensor (no pad)
    Returns two python lists of ints: z0, z1 of the same length
    Levenshtein-style DP (cost 1 for ins/del/sub, 0 for match)
    """
    L0 = seq0.size(0)
    L1 = seq1.size(0)

    # dp[i, j] = edit distance between seq0[:i] and seq1[:j]
    dp = torch.zeros((L0 + 1, L1 + 1), dtype=torch.long, device=seq0.device)
    for i in range(1, L0 + 1):
        dp[i, 0] = i
    for j in range(1, L1 + 1):
        dp[0, j] = j

    for i in range(1, L0 + 1):
        for j in range(1, L1 + 1):
            cost_sub = 0 if seq0[i-1].item() == seq1[j-1].item() else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,           # delete from seq0
                dp[i, j-1] + 1,           # insert into seq0
                dp[i-1, j-1] + cost_sub,  # substitute / match
            )

    # backtrace
    z0 = []
    z1 = []
    i, j = L0, L1
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost_sub = 0 if seq0[i-1].item() == seq1[j-1].item() else 1
            if dp[i, j].item() == dp[i-1, j-1].item() + cost_sub:
                # match or substitute
                z0.append(int(seq0[i-1].item()))
                z1.append(int(seq1[j-1].item()))
                i -= 1
                j -= 1
                continue
        # deletion from seq0
        if i > 0 and dp[i, j].item() == dp[i-1, j].item() + 1:
            z0.append(int(seq0[i-1].item()))
            z1.append(eps_id)
            i -= 1
            continue
        # insertion into seq0
        if j > 0 and dp[i, j].item() == dp[i, j-1].item() + 1:
            z0.append(eps_id)
            z1.append(int(seq1[j-1].item()))
            j -= 1
            continue

    # we built them backwards
    z0.reverse()
    z1.reverse()
    return z0, z1


def _suboptimal_align_1d(seq0, seq1, eps_id):
    """
    Simple left-aligned / monotonic alignment:
    pair i-th token with i-th token; if one runs out, pad with eps.
    seq0, seq1: 1D, no pad
    Returns python lists
    """
    L0 = seq0.size(0)
    L1 = seq1.size(0)
    N = max(L0, L1)
    z0, z1 = [], []
    for k in range(N):
        tok0 = int(seq0[k].item()) if k < L0 else eps_id
        tok1 = int(seq1[k].item()) if k < L1 else eps_id
        z0.append(tok0)
        z1.append(tok1)
    return z0, z1


def build_z0_z1_with_alignment(
    x0: torch.Tensor,  # (B, L0), padded with pad_id, contains BOS/EOS
    x1: torch.Tensor,  # (B, L1), padded with pad_id, contains BOS/EOS
    eps_id: int,
    pad_id: int,
    p_optimal: float = 0.6,
):
    """
    For each (x0[b], x1[b]) pair:
      - strip pad_id
      - keep BOS/EOS (they're part of the true sequence)
      - with prob p_optimal do DP alignment
      - else do suboptimal left-aligned alignment
    Then pad all (z0, z1) to the same length with eps_id.

    Returns:
      z0: (B, N_max)
      z1: (B, N_max)
    """
    device = x0.device
    B, L0 = x0.shape
    _, L1 = x1.shape

    z0_list = []
    z1_list = []
    max_len = 0

    # random numbers to pick alignment per sample
    rand = torch.rand(B, device=device)

    for b in range(B):
        # strip pads
        seq0 = x0[b][x0[b] != pad_id]  # 1D
        seq1 = x1[b][x1[b] != pad_id]  # 1D

        # choose alignment strategy
        if rand[b].item() < p_optimal:
            # most optimal (DP)
            aline0, aline1 = _optimal_align_1d(seq0, seq1, eps_id)
        else:
            # sub-optimal (monotonic)
            aline0, aline1 = _suboptimal_align_1d(seq0, seq1, eps_id)

        assert len(aline0) == len(aline1)
        cur_len = len(aline0)
        if cur_len > max_len:
            max_len = cur_len

        z0_list.append(aline0)
        z1_list.append(aline1)

    # now pad to max_len with eps_id
    z0 = torch.full((B, max_len), eps_id, dtype=torch.long, device=device)
    z1 = torch.full((B, max_len), eps_id, dtype=torch.long, device=device)

    for b in range(B):
        cur_len = len(z0_list[b])
        z0[b, :cur_len] = torch.tensor(z0_list[b], dtype=torch.long, device=device)
        z1[b, :cur_len] = torch.tensor(z1_list[b], dtype=torch.long, device=device)

    return z0, z1

def remove_eps(
    z_t: torch.Tensor,   # (B, N)
    eps_id: int,
    pad_id: int,
    return_mask: bool = True,
):
    device = z_t.device
    B, N = z_t.shape

    x_t = []
    for b in range(B):
        seq = z_t[b]
        core = seq[seq != eps_id]  # remove eps
        x_t.append(core)

    x_t = pad_sequence(x_t, batch_first=True, padding_value=pad_id)
    mask = (x_t != pad_id).bool()

    if return_mask:
        return x_t, mask
    return x_t