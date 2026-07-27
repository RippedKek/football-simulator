"""Stage 6 relational encoder -- the per-agent feature vector.

Raw world coordinates do not capture style; style is relational. This module
turns one team's positions into a per-agent vector encoding its own state plus
its relationship to the ball, the nearest opponent, the pressure around it, its
own team's shape, and -- for goalkeepers -- goal geometry.

Per-agent layout (24 dims), all normalised to ~unit scale:
   0-1   own position (x/105, y/68)
   2-3   own velocity (vx, vy / 10 m/s)
   4-6   vector to ball (dx/105, dy/68) and distance (/pitch diagonal)
   7-9   vector to nearest opponent and distance
   10    opponents within a 5 m pressure radius (count / 11)
   11-12 vector to own-team formation centroid
   13    own defensive-line height (/105)
   14-16 GK only: distance to own goal, angle subtended by the posts (/pi),
         nearest opponent's proximity to the penalty spot (zero for outfield)
   17-23 event label one-hot (7 classes; currently a 'none' stub)

Mirrored in torch by stage7_style.features_torch.build_frame -- keep the two in
sync.
"""

from __future__ import annotations

import numpy as np

N_SLOTS = 11
EVENT_CLASSES = ["none", "pass", "shot", "tackle", "clearance", "pressing", "set_piece"]
N_EVENTS = len(EVENT_CLASSES)
F_DIM = 17 + N_EVENTS       # 24
GOAL_HALF_W = 3.66          # metres, half of the 7.32 m goal
PEN_DIST = 11.0


def team_features(own_xy, own_v, other_xy, ball_xy, L, Wd, diag):
    """Relational features for one team's 11 agents over T frames -> (T,11,24).

    own_xy / own_v / other_xy in metres; slot 0 is the goalkeeper.
    """
    T = own_xy.shape[0]
    f = np.zeros((T, N_SLOTS, F_DIM), np.float32)

    f[..., 0] = own_xy[..., 0] / L
    f[..., 1] = own_xy[..., 1] / Wd
    f[..., 2] = own_v[..., 0] / 10.0
    f[..., 3] = own_v[..., 1] / 10.0

    # vector + distance to ball
    vb = ball_xy[:, None, :] - own_xy
    f[..., 4] = vb[..., 0] / L
    f[..., 5] = vb[..., 1] / Wd
    f[..., 6] = np.linalg.norm(vb, axis=-1) / diag

    # nearest opponent
    diff = own_xy[:, :, None, :] - other_xy[:, None, :, :]   # (T,11,11,2)
    d = np.linalg.norm(diff, axis=-1)                         # (T,11,11)
    j = np.argmin(d, axis=-1)                                 # (T,11)
    ti, ai = np.indices(j.shape)
    vno = -diff[ti, ai, j]                                    # vector own -> nearest opp
    f[..., 7] = vno[..., 0] / L
    f[..., 8] = vno[..., 1] / Wd
    f[..., 9] = d.min(axis=-1) / diag
    f[..., 10] = (d <= 5.0).sum(axis=-1) / N_SLOTS            # pressure count

    # formation centroid (outfield slots 1..10)
    cen = own_xy[:, 1:, :].mean(axis=1, keepdims=True)        # (T,1,2)
    vc = cen - own_xy
    f[..., 11] = vc[..., 0] / L
    f[..., 12] = vc[..., 1] / Wd

    # defensive-line height: oriented by GK side; mean x of the 4 deepest outfield
    gk_x = own_xy[:, 0, 0]
    goal_x = np.where(gk_x < L / 2, 0.0, L)                   # (T,)
    out_x = own_xy[:, 1:, 0]                                  # (T,10)
    depth = np.abs(out_x - goal_x[:, None])
    near4 = np.sort(depth, axis=1)[:, :4].mean(axis=1)        # (T,)
    f[..., 13] = (near4 / L)[:, None]

    # GK augmentation on slot 0
    gk = own_xy[:, 0, :]                                      # (T,2)
    gc = np.stack([goal_x, np.full(T, Wd / 2)], axis=1)       # goal centre
    f[:, 0, 14] = np.linalg.norm(gk - gc, axis=1) / L
    p_top = np.stack([goal_x, np.full(T, Wd / 2 - GOAL_HALF_W)], 1) - gk
    p_bot = np.stack([goal_x, np.full(T, Wd / 2 + GOAL_HALF_W)], 1) - gk
    cosang = (p_top * p_bot).sum(1) / (
        np.linalg.norm(p_top, axis=1) * np.linalg.norm(p_bot, axis=1) + 1e-6)
    f[:, 0, 15] = np.arccos(np.clip(cosang, -1, 1)) / np.pi
    pen_x = np.where(goal_x < L / 2, PEN_DIST, L - PEN_DIST)
    spot = np.stack([pen_x, np.full(T, Wd / 2)], axis=1)      # (T,2)
    opp_d = np.linalg.norm(other_xy - spot[:, None, :], axis=-1)   # (T,11)
    f[:, 0, 16] = opp_d.min(axis=1) / diag

    # event one-hot (stub: always 'none' -> index 0)
    f[..., 17] = 1.0
    return f
