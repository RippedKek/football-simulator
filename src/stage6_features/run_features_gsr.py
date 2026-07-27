"""Stage 6 (GSR variant) -- relational features for 23 agents (ball included).

Reuses the per-team relational encoder (`team_features`) for the 22 players and
adds a ball feature row (agent 22): position + velocity only, relational dims
zero -- the model distinguishes it via role/team embeddings and reads its
position through dims 0-1. Style vectors pass through unchanged.

Reads output/storage/gsr/windows_{train,val}.npz, writes
output/features/gsr/features_{train,val}.npz with agent_feat (N,30,23,24) plus
presence, target_delta, style0, style1.

    python -m stage6_features.run_features_gsr
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage6_features.relational import team_features, F_DIM, N_SLOTS

ROOT = Path(__file__).resolve().parents[2]
N_AGENTS = 23
BALL = 22


def ball_features(ball_xy, ball_v, L, Wd, bmax):
    """(T,F_DIM) for the ball: pos + velocity, relational dims zero, event onehot."""
    T = ball_xy.shape[0]
    f = np.zeros((T, F_DIM), np.float32)
    f[:, 0] = ball_xy[:, 0] / L
    f[:, 1] = ball_xy[:, 1] / Wd
    f[:, 2] = ball_v[:, 0] / bmax
    f[:, 3] = ball_v[:, 1] / bmax
    f[:, 17] = 1.0    # event one-hot stub -> 'none'
    return f


def process(npz, norm):
    L, Wd = norm["pos_x_m"], norm["pos_y_m"]
    diag = float(np.hypot(L, Wd))
    pmax, bmax = norm["player_max_speed_mps"], norm["ball_max_speed_mps"]
    agents = npz["agents"]
    keys = {"presence": npz["presence"], "target_delta": npz["target_delta"],
            "style0": npz["style0"], "style1": npz["style1"]}
    if agents.size == 0:
        return {"agent_feat": np.zeros((0,)), **keys}

    N, T = agents.shape[0], agents.shape[1]
    feats = np.zeros((N, T, N_AGENTS, F_DIM), np.float32)
    for n in range(N):
        a = agents[n]
        a_xy, a_v = a[:, :N_SLOTS, :2] * [L, Wd], a[:, :N_SLOTS, 2:4] * pmax
        o_xy, o_v = a[:, N_SLOTS:2 * N_SLOTS, :2] * [L, Wd], a[:, N_SLOTS:2 * N_SLOTS, 2:4] * pmax
        b_xy, b_v = a[:, BALL, :2] * [L, Wd], a[:, BALL, 2:4] * bmax
        feats[n, :, :N_SLOTS] = team_features(a_xy, a_v, o_xy, b_xy, L, Wd, diag)
        feats[n, :, N_SLOTS:2 * N_SLOTS] = team_features(o_xy, o_v, a_xy, b_xy, L, Wd, diag)
        feats[n, :, BALL] = ball_features(b_xy, b_v, L, Wd, bmax)
    return {"agent_feat": feats, **keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(ROOT / "output" / "storage" / "gsr"))
    ap.add_argument("--out", default=str(ROOT / "output" / "features" / "gsr"))
    args = ap.parse_args()
    store = Path(args.store)
    norm = json.loads((store / "meta.json").read_text())["normalisation"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ("train", "val"):
        npz = np.load(store / f"windows_{split}.npz")
        out = process(npz, norm)
        np.savez_compressed(out_dir / f"features_{split}.npz", **out)
        counts[split] = int(out["agent_feat"].shape[0]) if out["agent_feat"].size else 0

    fmeta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_dim": F_DIM, "n_agents": N_AGENTS,
        "agents_order": "0-10 team0 (0=GK), 11-21 team1 (11=GK), 22 ball",
        "train_windows": counts["train"], "val_windows": counts["val"],
    }
    (out_dir / "meta.json").write_text(json.dumps(fmeta, indent=2), encoding="utf-8")
    print(f"features train {counts['train']} / val {counts['val']} dim {F_DIM} "
          f"agents {N_AGENTS} -> {out_dir}")


if __name__ == "__main__":
    main()
