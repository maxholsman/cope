# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

from typing import List, Optional, Sequence
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

# If you kept the adapter around only for scheduler access you can drop it entirely,
# because we pass precomputed_weight from training.step. Keeping it is harmless though.
class EditFlowsLoss(_Loss):
    """
    Edit Flows loss (Eq. 23), ragged version:
      L_i =  (sum_j λ_ins[i][j] + sum_j λ_del[i][j] + sum_j λ_sub[i][j])
            - w_i * ( sum_{ins events e} [log λ_ins[i][slot_e] + log Q_ins[i](y_e)]
                    + sum_{del j}       [log λ_del[i][j]]
                    + sum_{sub j}       [log λ_sub[i][j] + log Q_sub[i](y_j)] )
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)

    def forward(
        self,
        lam_ins: torch.Tensor,     # (B, L)
        logits_ins: torch.Tensor,  # (B, L, V)
        lam_del: torch.Tensor,     # (B, L)
        lam_sub: torch.Tensor,     # (B, L)
        logits_sub: torch.Tensor,  # (B, L, V)
        z_t: torch.Tensor,         # (B, N) aligned, with eps_id
        z_1: torch.Tensor,         # (B, N) aligned target, with eps_id
        x_t: torch.Tensor,         # (B, L)
        valid_mask: torch.Tensor,  # (B, L) bool, False==padding, True==valid
        precomputed_weight: torch.Tensor,  # (B,) or ()  = kappa_dot/(1-kappa)
        eps_id: int,
        bos_id: int = 0,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """
        Implements Eq. 23 style loss for Edit Flows.

        We:
        1) penalize total outgoing rate on x_t
        2) for every column in (z_t, z_1) that still differs, we map it to an x_t edit
            and add  - w * log u_required

        BOS/EOS handling:
        - z_t and z_1 are aligned and already contain BOS/EOS
        - we do NOT allow edits that delete/replace BOS/EOS
        - we also skip columns whose target token is BOS/EOS (they should already match)
        """
        device = lam_ins.device
        B, L = x_t.shape
        _, N = z_t.shape

        # 1. ----- RATE TERM -----
        # valid_mask: True = real token, False = pad
        valid_f = valid_mask.to(lam_ins.dtype)  # (B, L) 1.0 for valid, 0.0 for pad

        # total outgoing rate at each position
        total_rate_pos = lam_ins + lam_del + lam_sub     # (B, L)
        total_rate_pos = total_rate_pos * valid_f        # zero out pads
        loss_rate = total_rate_pos.sum(dim=1)            # (B,)

        # 2. ----- EDIT TERM -----
        # precompute log-softmax for tokens (better numerics)
        logp_ins = F.log_softmax(logits_ins, dim=-1)   # (B, L, V)
        logp_sub = F.log_softmax(logits_sub, dim=-1)   # (B, L, V)

        # Make weight shape nice
        if precomputed_weight.dim() == 0:
            precomputed_weight = precomputed_weight.view(1).expand(B).to(device)
        else:
            precomputed_weight = precomputed_weight.to(device)

        # use float for accumulation
        loss_edit = torch.zeros(B, dtype=torch.float32, device=device)

        for b in range(B):
            # how many *valid* tokens in x_t[b]
            valid_len = int(valid_mask[b].sum().item())

            prefix_non_eps = 0  # = number of non-ε seen in z_t[b, :i]

            for i in range(N):
                zt = int(z_t[b, i].item())
                z1 = int(z_1[b, i].item())

                # map this column to an x_t position
                if zt != eps_id:
                    # this column corresponds to x_t[b, prefix_non_eps]
                    x_pos = prefix_non_eps
                    is_token = True
                    prefix_non_eps += 1
                else:
                    # this is a gap column, sits BETWEEN tokens
                    x_pos = prefix_non_eps
                    is_token = False

                # if already matched -> no term
                if zt == z1:
                    continue

                # --- BOS/EOS guards on the target side ---
                # if the target token is BOS or EOS, we shouldn't try to force an edit here
                if z1 == bos_id or z1 == eos_id:
                    # skip this column; aligned BOS/EOS should already match
                    continue

                # figure out which edit we need
                # CASE 1: deletion: z_t has token, z_1 has ε
                if is_token and (z1 == eps_id):
                    # delete token at x_pos
                    if x_pos >= valid_len:
                        # out of range (shouldn't happen if alignment & mask match)
                        raise NotImplementedError

                    x_token = int(x_t[b, x_pos].item())
                    # do NOT delete BOS/EOS
                    if x_token == bos_id or x_token == eos_id:
                        continue

                    lam = lam_del[b, x_pos].clamp_min(1e-12)
                    log_u_req = torch.log(lam)

                # CASE 2: substitution: token -> different token
                elif is_token and (z1 != eps_id) and (zt != z1):
                    if x_pos >= valid_len:
                        raise NotImplementedError

                    x_token = int(x_t[b, x_pos].item())
                    # do NOT substitute BOS/EOS
                    if x_token == bos_id or x_token == eos_id:
                        continue

                    lam = lam_sub[b, x_pos].clamp_min(1e-12)
                    logp_tok = logp_sub[b, x_pos, z1]
                    log_u_req = torch.log(lam) + logp_tok

                # CASE 3: insertion: ε -> token
                elif (not is_token) and (z1 != eps_id):
                    # insertion in the gap after token (x_pos - 1)
                    # if x_pos == 0, we insert after "BOS"/at start -> map to position 0
                    ins_pos = x_pos - 1
                    if ins_pos < 0:
                        ins_pos = 0

                    # clamp to last valid position if needed
                    if valid_len == 0:
                        # degenerate, but avoid -1
                        ins_pos = 0
                    elif ins_pos >= valid_len:
                        ins_pos = valid_len - 1

                    # also don't insert "after" EOS if ins_pos currently points to EOS
                    x_token = int(x_t[b, ins_pos].item())
                    if x_token == eos_id:
                        # simplest policy: skip this insertion supervision
                        continue

                    lam = lam_ins[b, ins_pos].clamp_min(1e-12)
                    logp_tok = logp_ins[b, ins_pos, z1]
                    log_u_req = torch.log(lam) + logp_tok

                else:
                    # unknown pattern (shouldn't happen with proper alignment)
                    raise NotImplementedError

                w = precomputed_weight[b]
                loss_edit[b] += - w * log_u_req

        # 3. ----- COMBINE -----
        loss = loss_rate + loss_edit  # (B,)
        loss = loss.mean()
        return loss
