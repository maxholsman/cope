from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

Tensor = torch.Tensor


def _optimal_align_core(core0: torch.Tensor, core1: torch.Tensor, eps_id: int):
    """
    Edit-distance alignment on the *core* (no BOS/EOS).
    Returns two python lists of ints of the same length, using eps_id for gaps.
    """
    L0 = core0.size(0)
    L1 = core1.size(0)

    dp = torch.zeros((L0 + 1, L1 + 1), dtype=torch.long, device=core0.device)
    for i in range(1, L0 + 1):
        dp[i, 0] = i
    for j in range(1, L1 + 1):
        dp[0, j] = j

    for i in range(1, L0 + 1):
        for j in range(1, L1 + 1):
            cost_sub = 0 if core0[i-1].item() == core1[j-1].item() else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,          # delete core0[i-1]
                dp[i, j-1] + 1,          # insert core1[j-1]
                dp[i-1, j-1] + cost_sub  # match/sub
            )

    z0_core = []
    z1_core = []
    i, j = L0, L1
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost_sub = 0 if core0[i-1].item() == core1[j-1].item() else 1
            if dp[i, j].item() == dp[i-1, j-1].item() + cost_sub:
                z0_core.append(int(core0[i-1].item()))
                z1_core.append(int(core1[j-1].item()))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j].item() == dp[i-1, j].item() + 1:
            z0_core.append(int(core0[i-1].item()))
            z1_core.append(eps_id)
            i -= 1
            continue
        if j > 0 and dp[i, j].item() == dp[i, j-1].item() + 1:
            z0_core.append(eps_id)
            z1_core.append(int(core1[j-1].item()))
            j -= 1
            continue

    z0_core.reverse()
    z1_core.reverse()
    return z0_core, z1_core


def _suboptimal_align_core(core0: torch.Tensor, core1: torch.Tensor, eps_id: int):
    """
    Left-align cores; pad the shorter core with eps_id.
    """
    L0 = core0.size(0)
    L1 = core1.size(0)
    N = max(L0, L1)
    z0_core, z1_core = [], []
    for k in range(N):
        tok0 = int(core0[k].item()) if k < L0 else eps_id
        tok1 = int(core1[k].item()) if k < L1 else eps_id
        z0_core.append(tok0)
        z1_core.append(tok1)
    return z0_core, z1_core


def build_z0_z1_with_alignment(
    x0: torch.Tensor,  # (B, L0), padded with pad_id, contains BOS/EOS
    x1: torch.Tensor,  # (B, L1), padded with pad_id, contains BOS/EOS
    eps_id: int,
    pad_id: int,
    bos_id: int,   
    eos_id: int,   
    p_optimal: float = 0.6,
):
    """
    Align x0 and x1 such that:
      - BOS aligns with BOS
      - EOS aligns with EOS
      - between BOS and EOS we align with eps_id
      - after EOS we pad with pad_id

    Returns:
      z0: (B, N_max)
      z1: (B, N_max)
    """
    device = x0.device
    B = x0.size(0)

    z0_list = []
    z1_list = []
    max_len = 0

    rand = torch.rand(B, device=device)

    for b in range(B):
        # strip pads
        seq0 = x0[b][x0[b] != pad_id]  # e.g. [BOS, ..., EOS]
        seq1 = x1[b][x1[b] != pad_id]

        # find BOS/EOS positions (assume 1 each, in order)
        # usually BOS is at index 0, but let's be safe
        bos_pos0 = (seq0 == bos_id).nonzero(as_tuple=False)[0, 0].item()
        bos_pos1 = (seq1 == bos_id).nonzero(as_tuple=False)[0, 0].item()
        eos_pos0 = (seq0 == eos_id).nonzero(as_tuple=False)[0, 0].item()
        eos_pos1 = (seq1 == eos_id).nonzero(as_tuple=False)[0, 0].item()

        # cores: everything between BOS and EOS
        core0 = seq0[bos_pos0 + 1 : eos_pos0]  # may be empty
        core1 = seq1[bos_pos1 + 1 : eos_pos1]

        # pick alignment strategy for the core
        if rand[b].item() < p_optimal:
            core0_aligned, core1_aligned = _optimal_align_core(core0, core1, eps_id)
        else:
            core0_aligned, core1_aligned = _suboptimal_align_core(core0, core1, eps_id)

        # rebuild full aligned sequences: [BOS] + core_aligned + [EOS]
        aligned0 = [bos_id] + core0_aligned + [eos_id]
        aligned1 = [bos_id] + core1_aligned + [eos_id]

        cur_len = len(aligned0)
        assert cur_len == len(aligned1)
        if cur_len > max_len:
            max_len = cur_len

        z0_list.append(aligned0)
        z1_list.append(aligned1)

    # pad with pad_id AFTER eos
    z0 = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    z1 = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)

    for b in range(B):
        cur = len(z0_list[b])
        z0[b, :cur] = torch.tensor(z0_list[b], device=device, dtype=torch.long)
        z1[b, :cur] = torch.tensor(z1_list[b], device=device, dtype=torch.long)

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