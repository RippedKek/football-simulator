"""Differentiable single-frame feature builder (torch port of Stage 6).

Mirrors stage6_features team_features / ball_features for one frame so a training
rollout can re-featurize the model's own predicted positions and keep gradients
flowing. Also reused by the simulator (M5) for consistency.

Input  agents: (B,23,4) NORMALISED (x/105, y/68, vx/pmax, vy/bmax-for-ball).
Output feats : (B,23,24) matching features_gsr.
"""

from __future__ import annotations

import torch

N_SLOTS = 11
BALL = 22
GOAL_HALF_W = 3.66
PEN_DIST = 11.0
FDIM = 24


def _team_feats(own_xy, own_v, other_xy, ball_xy, L, Wd, diag):
    B, N, _ = own_xy.shape
    f = torch.zeros(B, N, FDIM, device=own_xy.device, dtype=own_xy.dtype)
    f[..., 0] = own_xy[..., 0] / L
    f[..., 1] = own_xy[..., 1] / Wd
    f[..., 2] = own_v[..., 0] / 10.0
    f[..., 3] = own_v[..., 1] / 10.0

    vb = ball_xy[:, None, :] - own_xy
    f[..., 4] = vb[..., 0] / L
    f[..., 5] = vb[..., 1] / Wd
    f[..., 6] = vb.norm(dim=-1) / diag

    diff = own_xy[:, :, None, :] - other_xy[:, None, :, :]      # (B,N,N,2)
    d = diff.norm(dim=-1)                                       # (B,N,N)
    dmin, j = d.min(dim=-1)
    vno = -torch.gather(diff, 2, j[..., None, None].expand(B, N, 1, 2)).squeeze(2)
    f[..., 7] = vno[..., 0] / L
    f[..., 8] = vno[..., 1] / Wd
    f[..., 9] = dmin / diag
    f[..., 10] = (d <= 5.0).sum(-1).to(f.dtype) / N_SLOTS

    cen = own_xy[:, 1:, :].mean(dim=1, keepdim=True)
    vc = cen - own_xy
    f[..., 11] = vc[..., 0] / L
    f[..., 12] = vc[..., 1] / Wd

    gk_x = own_xy[:, 0, 0]
    goal_x = torch.where(gk_x < L / 2, torch.zeros_like(gk_x), torch.full_like(gk_x, L))
    out_x = own_xy[:, 1:, 0]
    depth = (out_x - goal_x[:, None]).abs()
    near4 = depth.sort(dim=1).values[:, :4].mean(dim=1)
    f[..., 13] = (near4 / L)[:, None]

    gk = own_xy[:, 0, :]
    half_w = torch.full_like(goal_x, Wd / 2)
    gc = torch.stack([goal_x, half_w], dim=1)
    f[:, 0, 14] = (gk - gc).norm(dim=1) / L
    p_top = torch.stack([goal_x, half_w - GOAL_HALF_W], 1) - gk
    p_bot = torch.stack([goal_x, half_w + GOAL_HALF_W], 1) - gk
    cosang = (p_top * p_bot).sum(1) / (p_top.norm(dim=1) * p_bot.norm(dim=1) + 1e-6)
    f[:, 0, 15] = torch.arccos(cosang.clamp(-1, 1)) / torch.pi
    pen_x = torch.where(goal_x < L / 2, torch.full_like(goal_x, PEN_DIST),
                        torch.full_like(goal_x, L - PEN_DIST))
    spot = torch.stack([pen_x, half_w], dim=1)
    opp_d = (other_xy - spot[:, None, :]).norm(dim=-1)
    f[:, 0, 16] = opp_d.min(dim=1).values / diag

    f[..., 17] = 1.0
    return f


def build_frame(agents, norm):
    """agents (B,23,4) normalised -> feats (B,23,24)."""
    L, Wd = norm["pos_x_m"], norm["pos_y_m"]
    pmax, bmax = norm["player_max_speed_mps"], norm["ball_max_speed_mps"]
    diag = float((L ** 2 + Wd ** 2) ** 0.5)
    scale_xy = torch.tensor([L, Wd], device=agents.device, dtype=agents.dtype)

    a0 = agents[:, :N_SLOTS]
    a1 = agents[:, N_SLOTS:BALL]
    ab = agents[:, BALL]
    a0_xy, a0_v = a0[..., :2] * scale_xy, a0[..., 2:4] * pmax
    a1_xy, a1_v = a1[..., :2] * scale_xy, a1[..., 2:4] * pmax
    b_xy, b_v = ab[:, :2] * scale_xy, ab[:, 2:4] * bmax

    B = agents.shape[0]
    feats = torch.zeros(B, 23, FDIM, device=agents.device, dtype=agents.dtype)
    feats[:, :N_SLOTS] = _team_feats(a0_xy, a0_v, a1_xy, b_xy, L, Wd, diag)
    feats[:, N_SLOTS:BALL] = _team_feats(a1_xy, a1_v, a0_xy, b_xy, L, Wd, diag)
    feats[:, BALL, 0] = b_xy[:, 0] / L
    feats[:, BALL, 1] = b_xy[:, 1] / Wd
    feats[:, BALL, 2] = b_v[:, 0] / bmax
    feats[:, BALL, 3] = b_v[:, 1] / bmax
    feats[:, BALL, 17] = 1.0
    return feats
