"""KickNet -- pass/shoot/hold decision + receiver selection (hybrid engine).

Pointer-style: a shared MLP embeds each teammate relative to the kicker's
context; per-teammate scores are the receiver logits, and the masked mean of
the embeddings feeds the 3-way action head (hold / pass / shot).

Inputs (see build_kick_data):
  ctx (B,8)  mates (B,11,6)  mask (B,11)   -- all in the kicker's attack frame
Outputs:
  action_logits (B,3), receiver_logits (B,11) (absent/self slots at -1e4)
"""

from __future__ import annotations

import torch
import torch.nn as nn

CTX_DIM, MATE_DIM, N_SLOTS = 8, 6, 11


class KickNet(nn.Module):
    def __init__(self, d: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mate_mlp = nn.Sequential(
            nn.Linear(CTX_DIM + MATE_DIM, d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.GELU())
        self.score = nn.Linear(d, 1)
        self.action = nn.Sequential(
            nn.Linear(CTX_DIM + d, d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, 3))

    def forward(self, ctx, mates, mask):
        B = ctx.shape[0]
        x = torch.cat([ctx[:, None].expand(B, N_SLOTS, CTX_DIM), mates], -1)
        h = self.mate_mlp(x)                                   # (B,11,d)
        recv = self.score(h).squeeze(-1) + (mask - 1.0) * 1e4  # mask absent/self
        denom = mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask[..., None]).sum(1) / denom
        act = self.action(torch.cat([ctx, pooled], -1))
        return act, recv
