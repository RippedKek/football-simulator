"""M3 -- StyleNetGSR: style-controllable spatial-temporal transformer.

Changes from the original StyleNet (docs/01, docs/06):
  * 23 agents (22 players + ball); role/team embeddings gain a ball class.
  * Trajectory head predicts all 23 deltas, so the ball has learned dynamics.
  * **FiLM conditioning**: the four style knobs are an *input*. A small MLP maps
    each agent's team style vector to per-agent (gamma, beta) that modulate the
    embedding, so style drives behaviour. (The old model only read style out.)
  * Stacked N blocks of spatial(agents)->temporal(time, causal)->FF.
  * A style-consistency head predicts the knobs back from the pooled embedding;
    its loss forces the knobs to actually influence the trajectory (M4).

forward returns: delta(B,T,23,2), style_pred(B,2,4), event(B,7)[, attn].
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_SLOTS = 11
N_AGENTS = 23
BALL = 22


class Block(nn.Module):
    def __init__(self, D, heads, dropout):
        super().__init__()
        self.spatial = nn.MultiheadAttention(D, heads, dropout=dropout, batch_first=True)
        self.temporal = nn.MultiheadAttention(D, heads, dropout=dropout, batch_first=True)
        self.sp_norm = nn.LayerNorm(D)
        self.tp_norm = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))
        self.ff_norm = nn.LayerNorm(D)

    def forward(self, h, kpm, causal, return_attn):
        B, T, A, D = h.shape
        x = h.reshape(B * T, A, D)
        a, attn = self.spatial(x, x, x, key_padding_mask=kpm,
                               need_weights=return_attn, average_attn_weights=True)
        h = self.sp_norm(x + a).reshape(B, T, A, D)
        sp_attn = attn.reshape(B, T, A, A) if return_attn else None

        y = h.permute(0, 2, 1, 3).reshape(B * A, T, D)
        t, _ = self.temporal(y, y, y, attn_mask=causal)
        y = self.tp_norm(y + t).reshape(B, A, T, D).permute(0, 2, 1, 3)
        h = self.ff_norm(y + self.ff(y))
        return h, sp_attn


class StyleNetGSR(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        D = m["embed_dim"]
        self.D = D
        self.n_agents = m["n_agents"]
        self.n_style = m["n_style"]
        self.encoder = nn.Sequential(nn.Linear(m["feature_dim"], D), nn.GELU(), nn.LayerNorm(D))
        self.role_emb = nn.Embedding(3, D)   # 0 GK, 1 outfield, 2 ball
        self.team_emb = nn.Embedding(3, D)   # 0 team0, 1 team1, 2 ball
        self.film = nn.Sequential(nn.Linear(self.n_style, D), nn.GELU(), nn.Linear(D, 2 * D))
        self.blocks = nn.ModuleList([Block(D, m["n_heads"], m["dropout"])
                                     for _ in range(m["n_blocks"])])
        self.traj_head = nn.Linear(D, 2)
        self.style_head = nn.Linear(D, self.n_style)
        self.event_head = nn.Linear(D, 7)

        roles = torch.ones(N_AGENTS, dtype=torch.long)
        roles[0] = 0; roles[N_SLOTS] = 0; roles[BALL] = 2
        teams = torch.zeros(N_AGENTS, dtype=torch.long)
        teams[N_SLOTS:BALL] = 1; teams[BALL] = 2
        self.register_buffer("roles", roles)
        self.register_buffer("teams", teams)

    def _agent_style(self, style0, style1):
        B = style0.shape[0]
        s = torch.empty(B, N_AGENTS, self.n_style, device=style0.device, dtype=style0.dtype)
        s[:, :N_SLOTS] = style0[:, None]
        s[:, N_SLOTS:BALL] = style1[:, None]
        s[:, BALL] = 0.5 * (style0 + style1)
        return s

    def forward(self, feat, presence, style0, style1, mask_thr=0.2, return_attn=False):
        B, T, A, _ = feat.shape
        h = self.encoder(feat)
        h = h + self.role_emb(self.roles)[None, None] + self.team_emb(self.teams)[None, None]

        gamma, beta = self.film(self._agent_style(style0, style1)).chunk(2, dim=-1)  # (B,A,D)
        h = h * (1 + gamma[:, None]) + beta[:, None]

        kpm = presence.reshape(B * T, A) <= mask_thr
        kpm[kpm.all(dim=-1)] = False
        causal = torch.triu(torch.ones(T, T, device=feat.device, dtype=torch.bool), 1)

        attn = None
        for blk in self.blocks:
            h, a = blk(h, kpm, causal, return_attn)
            if a is not None:
                attn = a

        delta = self.traj_head(h)
        team0 = h[:, :, :N_SLOTS].mean(dim=(1, 2))
        team1 = h[:, :, N_SLOTS:BALL].mean(dim=(1, 2))
        style_pred = torch.stack([self.style_head(team0), self.style_head(team1)], dim=1)
        event_logits = self.event_head(team0)
        if return_attn:
            return delta, style_pred, event_logits, attn
        return delta, style_pred, event_logits
