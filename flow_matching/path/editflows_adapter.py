from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple

import torch
from torch import Tensor

from flow_matching.path import MixtureDiscreteProbPath


@dataclass
class EditFlowsPathSampleRagged:
    # aligned inputs/outputs (ragged)
    z0_list: List[Tensor]      # list[(T_i,)]
    z1_list: List[Tensor]      # list[(T_i,)]
    zt_list: List[Tensor]      # list[(T_i,)]
    # projected current sequences (ragged X_t)
    x_t_list: List[Tensor]     # list[(n_i,)]
    # targets/masks for EF loss (ragged)
    need_delete_list: List[Tensor]        # list[(n_i,)]   bool
    need_substitute_list: List[Tensor]    # list[(n_i,)]   bool
    sub_target_list: List[Tensor]         # list[(n_i,)]   long
    ins_slot_idx_list: List[Tensor]       # list[(K_i,)]   long  indices in [0..n_i]
    ins_target_list: List[Tensor]         # list[(K_i,)]   long
    # time
    t: Tensor                              # (B,)


class EditFlowsPathAdapter:
    """
    Ragged adapter for Edit Flows:
      - Accepts aligned Z sequences as lists (may include eps_id).
      - Applies MixtureDiscreteProbPath's scheduler (σ_t) to sample z_t by
        flipping z0→z1 with prob (1-σ_t) per aligned token.
      - Projects z_t to x_t by removing ε.
      - Builds EF loss targets as ragged structures:
          * deletions/substitutions: per-token masks (length n_i)
          * insertions: per-event lists (slot_idx, target) allowing multiple per slot
    """

    def __init__(self, mixture_path: MixtureDiscreteProbPath, eps_id: int):
        self.path = mixture_path
        self.eps_id = int(eps_id)

    # expose scheduler so training can precompute weights κ̇/(1-κ)
    def scheduler(self, t: Tensor):
        return self.path.scheduler(t)

    @torch.no_grad()
    def sample(
        self,
        *,
        z0: Sequence[Tensor],   # list[(T_i,)]
        z1: Sequence[Tensor],   # list[(T_i,)]
        t:  Tensor,             # (B,)
    ) -> EditFlowsPathSampleRagged:
        assert isinstance(z0, (list, tuple)) and isinstance(z1, (list, tuple)), \
            "EditFlowsPathAdapter.sample expects ragged lists for z0,z1"
        B = len(z0)
        assert len(z1) == B and t.numel() == B, "Batch sizes must match"
        if B == 0:
            return EditFlowsPathSampleRagged([], [], [], [], [], [], [], [], [], t)

        device = z0[0].device
        lengths = torch.tensor([zi.numel() for zi in z0], device=device, dtype=torch.long)
        assert all(z0[i].numel() == z1[i].numel() for i in range(B)), "Aligned z0,z1 must share lengths"

        # ---- 1) Mixture flip in flattened space ----
        flat0 = torch.cat(z0, dim=0) if lengths.sum().item() > 0 else torch.empty(0, dtype=torch.long, device=device)
        flat1 = torch.cat(z1, dim=0) if lengths.sum().item() > 0 else torch.empty(0, dtype=torch.long, device=device)

        sched = self.path.scheduler(t)           # has sigma_t (prob to keep z0)
        sigma_b = sched.sigma_t                  # (B,)
        sigma_flat = torch.repeat_interleave(sigma_b, lengths)  # (sum T_i,)

        keep = (torch.rand_like(sigma_flat) < sigma_flat)       # (sum T_i,)
        flat_t = torch.where(keep, flat0, flat1)

        # back to ragged aligned z_t
        splits = lengths.tolist()
        zt_list = list(torch.split(flat_t, splits))

        # ---- 2) Project z_t → x_t by removing ε ----
        x_t_list: List[Tensor] = []
        for zt in zt_list:
            if zt.numel() == 0:
                x_t_list.append(zt)               # empty
            else:
                x_t_list.append(zt[zt != self.eps_id])

        # ---- 3) Build EF targets from alignment (z_t, z_1) ----
        # We scan aligned pairs and maintain a running token index into x_t.
        need_delete_list: List[Tensor] = []
        need_subst_list:  List[Tensor] = []
        sub_target_list:  List[Tensor] = []
        ins_slot_idx_list: List[Tensor] = []
        ins_target_list:   List[Tensor] = []

        for i in range(B):
            zt = zt_list[i]
            z1_i = z1[i]
            n_i = x_t_list[i].numel()

            # Per-token masks (length n_i)
            del_mask = torch.zeros(n_i, dtype=torch.bool, device=device)
            sub_mask = torch.zeros(n_i, dtype=torch.bool, device=device)
            sub_tgt  = torch.zeros(n_i, dtype=torch.long, device=device)

            # Insertion events
            ins_slots: List[int] = []
            ins_tgts:  List[int] = []

            # running count of non-eps tokens in z_t gives:
            #   - current token index (for delete/sub)
            #   - current slot index (for insert)
            token_idx = 0
            for k in range(zt.numel()):
                a = int(zt[k].item())
                b = int(z1_i[k].item())

                if a == self.eps_id and b != self.eps_id:
                    # insertion at the current slot index = token_idx
                    ins_slots.append(token_idx)
                    ins_tgts.append(b)
                    # token_idx unchanged
                elif a != self.eps_id and b == self.eps_id:
                    # deletion of the current token
                    if token_idx < n_i:  # guard for safety
                        del_mask[token_idx] = True
                    token_idx += 1
                elif a != self.eps_id and b != self.eps_id:
                    # substitution if different
                    if token_idx < n_i and a != b:
                        sub_mask[token_idx] = True
                        sub_tgt[token_idx]  = b
                    token_idx += 1
                else:
                    # a == ε and b == ε : no-op, token_idx unchanged
                    pass

            need_delete_list.append(del_mask)
            need_subst_list.append(sub_mask)
            sub_target_list.append(sub_tgt)

            if len(ins_slots) > 0:
                ins_slot_idx_list.append(torch.tensor(ins_slots, dtype=torch.long, device=device))
                ins_target_list.append(torch.tensor(ins_tgts, dtype=torch.long, device=device))
            else:
                ins_slot_idx_list.append(torch.empty(0, dtype=torch.long, device=device))
                ins_target_list.append(torch.empty(0, dtype=torch.long, device=device))

        return EditFlowsPathSampleRagged(
            z0_list=list(z0),
            z1_list=list(z1),
            zt_list=zt_list,
            x_t_list=x_t_list,
            need_delete_list=need_delete_list,
            need_substitute_list=need_subst_list,
            sub_target_list=sub_target_list,
            ins_slot_idx_list=ins_slot_idx_list,
            ins_target_list=ins_target_list,
            t=t,
        )
