"""Stage 5 shared helpers: velocity smoothing, speed clamping, dense arrays.

Used by run_storage_metrica. Kept separate so the windowing script stays about
windowing.

Agent order everywhere below: 0-10 team0 (0 = GK), 11-21 team1 (11 = GK),
22 = ball.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

N_SLOTS = 11
N_AGENTS = 23
BALL = 22


def smooth_velocities(df: pd.DataFrame, fps: float, wl: int, poly: int) -> pd.DataFrame:
    """Recompute velocities from Savitzky-Golay smoothed positions.

    Grouped per (segment, team, slot) so a track is never smoothed across a
    half-time boundary or between two different players sharing a slot.
    """
    df = df.sort_values(["segment_id", "team_id", "slot_id", "frame_index"]).copy()
    out = []
    for _, g in df.groupby(["segment_id", "team_id", "slot_id"], sort=False):
        g = g.copy()
        n = len(g)
        if n >= 5:
            w = min(wl, n if n % 2 == 1 else n - 1)
            w = max(5, w if w % 2 == 1 else w - 1)
            p = min(poly, w - 1)
            sx = savgol_filter(g.x_world.values, w, p)
            sy = savgol_filter(g.y_world.values, w, p)
        else:
            sx, sy = g.x_world.values, g.y_world.values
        g["x_s"], g["y_s"] = sx, sy
        t = g.frame_index.values / fps
        vx = np.gradient(sx, t) if n > 1 else np.zeros(n)
        vy = np.gradient(sy, t) if n > 1 else np.zeros(n)
        g["vx"], g["vy"] = vx, vy
        out.append(g)
    return pd.concat(out, ignore_index=True)


def clamp_speed(df: pd.DataFrame, player_max: float, ball_max: float) -> pd.DataFrame:
    """Re-clamp velocity after smoothing/np.gradient re-introduce teleports."""
    sp = np.hypot(df.vx.values, df.vy.values)
    cap = np.where(df["class"].values == "ball", ball_max, player_max)
    scale = np.where(sp > cap, cap / np.maximum(sp, 1e-6), 1.0)
    df = df.copy()
    df["vx"] *= scale
    df["vy"] *= scale
    return df


def segment_arrays(seg_df: pd.DataFrame, frames):
    """Long-format rows -> dense agents (F,23,4) and presence (F,23).

    The 4 channels are (x, y, vx, vy) in metres / m per second (not yet
    normalised). Expects the class column renamed to `class_` so it survives
    itertuples().
    """
    idx = {f: i for i, f in enumerate(frames)}
    F = len(frames)
    agents = np.zeros((F, N_AGENTS, 4), np.float32)
    pres = np.zeros((F, N_AGENTS), np.float32)
    sub = seg_df[seg_df.frame_index.isin(idx)]
    for r in sub.itertuples():
        i = idx[r.frame_index]
        feat = (r.x_world, r.y_world, r.vx, r.vy)
        if r.class_ == "ball":
            agents[i, BALL] = feat
            pres[i, BALL] = r.presence_weight
        elif r.team_id == 0:
            agents[i, r.slot_id] = feat
            pres[i, r.slot_id] = r.presence_weight
        elif r.team_id == 1:
            agents[i, N_SLOTS + r.slot_id] = feat
            pres[i, N_SLOTS + r.slot_id] = r.presence_weight
    return agents, pres


def normalise(agents, nrm):
    """Positions to [0,1] by pitch size; velocities by the per-class max speed."""
    a = agents.copy()
    a[..., 0] /= nrm["pos_x_m"]
    a[..., 1] /= nrm["pos_y_m"]
    a[:, :BALL, 2:4] /= nrm["player_max_speed_mps"]
    a[:, BALL:, 2:4] /= nrm["ball_max_speed_mps"]
    return a
