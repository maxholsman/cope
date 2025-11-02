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
            - w_i * ( sum_{ins j} [log λ_ins[i][j] + log Q_ins[i](target_j)]
                    + sum_{del j} [log λ_del[i][j]]
                    + sum_{sub j} [log λ_sub[i][j] + log Q_sub[i](target_j)] )
    where w_i = κ̇(t_i) / (1 - κ(t_i)). All inputs are lists; no padding masks needed.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)

    @staticmethod
    def _log_softmax_list(logits_list: Sequence[Tensor]) -> List[Tensor]:
        # logits: [(n_i(+1), V)] -> log_probs per sample
        return [F.log_softmax(lgts, dim=-1) if lgts.numel() > 0 else lgts for lgts in logits_list]

    def forward(
        self,
        *,
        # === model outputs (ragged lists) ===
        lam_ins: Sequence[Tensor],     # list[(n_i+1,)]
        lam_del: Sequence[Tensor],     # list[(n_i,)]
        lam_sub: Sequence[Tensor],     # list[(n_i,)]
        logits_ins: Sequence[Tensor],  # list[(n_i+1, V)]  (logits)
        logits_sub: Sequence[Tensor],  # list[(n_i,   V)]  (logits)
        # === targets/masks (ragged lists) ===
        need_delete: Sequence[Tensor],         # list[(n_i,  )] bool
        need_substitute: Sequence[Tensor],     # list[(n_i,  )] bool
        ins_slot_idx: Sequence[Tensor],        # list[(K_i,)] long, values in [0..n_i]
        ins_target:   Sequence[Tensor],        # list[(K_i,)] long
        sub_target:   Sequence[Tensor],        # list[(n_i,  )] long
        # === times ===
        t: Tensor,                              # (B,)  (unused if precomputed_weight is provided)
        precomputed_weight: Optional[Tensor] = None,   # (B,)
    ) -> Tensor:
        """
        Ragged Edit Flows loss (Eq. 23) with event-based insertions.
        Computes, per sample i:
            L_i =  sum_j [λ_ins^i[j]] + sum_j [λ_del^i[j]] + sum_j [λ_sub^i[j]]
                   - w_i * ( Σ_e [log λ_ins^i[s_e] + log Q_ins^i[s_e](y_e)]
                             + Σ_{j∈Del} [log λ_del^i[j]]
                             + Σ_{j∈Sub} [log λ_sub^i[j] + log Q_sub^i[j](y_j)] )
        where w_i = κ̇(t_i) / (1 - κ(t_i)).
        """
        B = len(lam_del)
        # Basic consistency checks across ragged lists
        assert all(len(lst) == B for lst in (
            lam_ins, lam_sub, logits_ins, logits_sub,
            need_delete, need_substitute, ins_slot_idx, ins_target, sub_target
        )), "All ragged inputs must have the same batch length"

        if precomputed_weight is None:
            raise ValueError("EditFlowsLoss expects `precomputed_weight` when used in ragged mode.")

        # Prepare log-softmax distributions (per sample)
        logQ_ins_list = self._log_softmax_list(logits_ins)  # [(n_i+1, V)]
        logQ_sub_list = self._log_softmax_list(logits_sub)  # [(n_i,   V)]

        device = precomputed_weight.device
        loss_per_sample = torch.zeros(B, device=device, dtype=torch.float32)

        for i in range(B):
            # ------- Term A: sum of outgoing rates -------
            tA = torch.tensor(0.0, device=device)
            if lam_ins[i].numel() > 0:
                tA = tA + lam_ins[i].sum()
            if lam_del[i].numel() > 0:
                tA = tA + lam_del[i].sum()
            if lam_sub[i].numel() > 0:
                tA = tA + lam_sub[i].sum()

            # ------- Term B: weighted log-likelihood over remaining edits -------
            tB = torch.tensor(0.0, device=device)

            # Insertions: event-based (multiple per slot allowed)
            if ins_slot_idx[i].numel() > 0:
                # print(f"lam_ins:                {lam_ins[i].shape}")
                # print(f"logQ_ins_list:          {logQ_ins_list[i].shape}")
                li = torch.clamp(lam_ins[i], min=1e-20)            # (n_i+1,)
                lp_ins = li.log()                                  # (n_i+1,)
                lq_ins = logQ_ins_list[i]                          # (n_i+1, V)
            

                sl = ins_slot_idx[i]                               # (K_i,)
                tg = ins_target[i]                                 # (K_i,)
                # print(f"li.shape: {li.shape}")
                # print(f"lp_ins.shape: {lp_ins.shape}")
                # print(f"lq_ins.shape: {lq_ins.shape}")
                # print(f"sl.shape: {sl.shape}")
                # print(f"tg.shape: {tg.shape}")

                lp_e = lp_ins.index_select(0, sl)                  # (K_i,)
                # print(f"lp_e.shape: {lp_e.shape}")

                lq_sl = lq_ins.index_select(0, sl)
                # print(f"lq_sl.shape: {lq_sl.shape}")

                lq_e = lq_sl.gather(1, tg.unsqueeze(1)).squeeze(1)# (K_i,)
                # print(f"lq_e.shape: {lq_e.shape}")
                tB = tB + (lp_e + lq_e).sum()
                # print(f"--------------------------------")

            # Deletions: per-token mask (each token can be deleted at most once)
            if need_delete[i].any():
                ld = torch.clamp(lam_del[i], min=1e-20)            # (n_i,)
                lp_del = ld.log()                                  # (n_i,)
                del_mask = need_delete[i].to(lp_del.dtype)         # (n_i,)
                tB = tB + (lp_del * del_mask).sum()

            # Substitutions: per-token mask
            if need_substitute[i].any():
                ls = torch.clamp(lam_sub[i], min=1e-20)            # (n_i,)
                lp_sub = ls.log()                                  # (n_i,)
                lq_sub = logQ_sub_list[i]                          # (n_i, V)
                tgt_sub = lq_sub.gather(1, sub_target[i].unsqueeze(1)).squeeze(1)  # (n_i,)
                sub_mask = need_substitute[i].to(lp_sub.dtype)     # (n_i,)
                tB = tB + ((lp_sub + tgt_sub) * sub_mask).sum()

            # Combine with per-sample weight
            w_i = precomputed_weight[i]
            loss_per_sample[i] = tA - w_i * tB

        # Reduction
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        elif self.reduction == "none":
            return loss_per_sample
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")



# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the CC-by-NC license found in the
# # LICENSE file in the root directory of this source tree.

# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from typing import Optional
# from torch.nn.modules.loss import _Loss

# from flow_matching.path import MixtureDiscreteProbPath, EditFlowsPathAdapter # for scheduler (κ, κdot)

# class EditFlowsLoss(_Loss):
#     """
#     Edit Flows loss (Eq. 23): sum of outgoing edit rates minus a weighted
#     log-likelihood over the remaining edits between z_t and z_1.
#     Weight is κdot / (1 - κ). See Fig. 3 and Sec. 3.2 in the paper.
#     """

#     def __init__(self, path: EditFlowsPathAdapter, reduction: str = "mean") -> None:
#         super().__init__(None, None, reduction)
#         self.path = path  # provides scheduler: alpha_t ≡ κ_t, d_alpha_t ≡ κdot_t

#     @staticmethod
#     def _safe_log_softmax(logits: Tensor, dim: int) -> Tensor:
#         return F.log_softmax(logits, dim=dim)

#     def forward(
#         self,
#         *,
#         # === model outputs ===
#         lam_ins: Tensor,          # (B, n+1)  ≥0
#         lam_del: Tensor,          # (B, n)    ≥0
#         lam_sub: Tensor,          # (B, n)    ≥0
#         logits_ins: Tensor,       # (B, n+1, V)
#         logits_sub: Tensor,       # (B, n, V)
#         # === alignment/targets (from z_t,z_1) ===
#         need_insert: Tensor,      # (B, n+1) bool — where z_t has ε and z_1 has token
#         need_delete: Tensor,      # (B, n)   bool — where z_t has token and z_1 has ε
#         need_substitute: Tensor,  # (B, n)   bool — where z_t token ≠ z_1 token (both non-ε)
#         ins_target: Tensor,       # (B, n+1) int token ids; ignored where need_insert==0
#         sub_target: Tensor,       # (B, n)   int token ids; ignored where need_substitute==0
#         # === validity masks per sample ===
#         token_valid: Tensor,      # (B, n)   bool — which token positions exist in x_t
#         slot_valid: Tensor,       # (B, n+1) bool — which insertion slots exist in x_t
#         # === time ===
#         t: Tensor,                # (B,)
#         precomputed_weight: Optional[Tensor] = None
#     ) -> Tensor:

#         B = lam_del.size(0)

#         if precomputed_weight is None:
#             # Scheduler values (α ≡ κ)
#             sched = self.path.scheduler(t)  # provides attributes alpha_t, d_alpha_t
#             weight = (sched.d_alpha_t / (1.0 - sched.alpha_t)).view(B, *([1] * (lam_del.dim() - 1)))
#         else:
#             weight = precomputed_weight

#         # -----------------------------
#         # Term A: sum of all outgoing rates (valid positions only)
#         # -----------------------------
#         termA_ins = torch.sum(lam_ins * slot_valid.to(lam_ins.dtype), dim=-1)   # (B,)
#         termA_del = torch.sum(lam_del * token_valid.to(lam_del.dtype), dim=-1)  # (B,)
#         termA_sub = torch.sum(lam_sub * token_valid.to(lam_sub.dtype), dim=-1)  # (B,)
#         termA = termA_ins + termA_del + termA_sub                               # (B,)

#         # -----------------------------
#         # Term B: log-likelihood over remaining edits
#         #   Insert:   log λ_ins + log Q_ins(target)
#         #   Delete:   log λ_del
#         #   Substit.: log λ_sub + log Q_sub(target)
#         # -----------------------------
#         logQ_ins = self._safe_log_softmax(logits_ins, dim=-1)  # (B, n+1, V)
#         logQ_sub = self._safe_log_softmax(logits_sub, dim=-1)  # (B, n, V)

#         # Gathers (mask will zero-out unused)
#         ins_logprob = torch.zeros_like(termA_ins)
#         if need_insert.any():
#             li = torch.clamp(lam_ins, min=1e-20)
#             lp_ins = torch.log(li)
#             tgt_ins = torch.gather(logQ_ins, dim=-1, index=ins_target.unsqueeze(-1)).squeeze(-1)
#             ins_term = (lp_ins + tgt_ins) * need_insert.to(lp_ins.dtype) * slot_valid.to(lp_ins.dtype)
#             ins_logprob = ins_term.sum(dim=-1)  # (B,)

#         del_logprob = torch.zeros_like(termA_del)
#         if need_delete.any():
#             ld = torch.clamp(lam_del, min=1e-20)
#             lp_del = torch.log(ld)
#             del_term = lp_del * need_delete.to(lp_del.dtype) * token_valid.to(lp_del.dtype)
#             del_logprob = del_term.sum(dim=-1)  # (B,)

#         sub_logprob = torch.zeros_like(termA_sub)
#         if need_substitute.any():
#             ls = torch.clamp(lam_sub, min=1e-20)
#             lp_sub = torch.log(ls)
#             tgt_sub = torch.gather(logQ_sub, dim=-1, index=sub_target.unsqueeze(-1)).squeeze(-1)
#             sub_term = (lp_sub + tgt_sub) * need_substitute.to(lp_sub.dtype) * token_valid.to(lp_sub.dtype)
#             sub_logprob = sub_term.sum(dim=-1)  # (B,)

#         termB = ins_logprob + del_logprob + sub_logprob  # (B,)

#         # -----------------------------
#         # Final scalar loss
#         # -----------------------------
#         loss_per_sample = termA - weight.squeeze() * termB  # (B,)

#         if self.reduction == "mean":
#             return loss_per_sample.mean()
#         elif self.reduction == "sum":
#             return loss_per_sample.sum()
#         elif self.reduction == "none":
#             return loss_per_sample
#         else:
#             raise ValueError(f"{self.reduction} is not a valid value for reduction")
