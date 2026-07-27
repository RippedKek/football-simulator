"""M1 -- interpretable style descriptors (the four tunable knobs).

From a clip's tracks.parquet we compute, per team, four style values in [0,1]:

  line_height  high line vs deep block   (full-team, needs visibility)
  press        tight close-down vs sit   (ball-local, robust)
  width        stretched vs narrow shape  (full-team, needs visibility)
  tempo        fast & vertical vs slow    (ball-local, robust)

Full-team metrics (line_height, width) are computed only on frames with at least
MIN_VIS visible outfielders, then aggregated by median over the clip -- because
broadcast shows ~7/team on average (see docs/02, docs/03). Ball-local metrics
(press, tempo) use every frame the ball is present.

Coordinates are oriented to each team's *attacking direction* (inferred from its
goalkeeper's mean x): forward = toward the goal the team attacks.

    python -m stage_style.descriptors --all      # writes style.json per clip + summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PITCH_L, PITCH_W = 105.0, 68.0

MIN_VIS = 6           # min visible outfielders for full-team metrics
POSS_R = 5.0          # m: player within this of ball can be possessor
PRESS_NORM = 15.0     # m: ball-to-nearest-defender distance that maps to press=0
BALL_SPEED_NORM = 8.0 # m/s: ball speed mapping to tempo=1
DEEP_K = 4            # how many deepest defenders define the back line


def attack_dir(players: pd.DataFrame, team: int) -> float:
    """+1 if team attacks toward x=105, -1 toward x=0. Inferred from GK (else mean x)."""
    gk = players[(players.team_id == team) & (players.role_id == 0)].x_world
    ref = gk.mean() if len(gk) else players[players.team_id == team].x_world.mean()
    return 1.0 if ref < PITCH_L / 2 else -1.0


def _fwd(x, d):
    return x if d > 0 else PITCH_L - x


def compute_style(df: pd.DataFrame, max_gap: int = 3) -> dict:
    """Per-team style dict. Segment-aware: attack direction is inferred PER
    SEGMENT (teams swap ends at half-time and Metrica does not flip coords),
    and possession runs never bridge a segment boundary.

    max_gap: max frame-index gap treated as contiguous in a possession run --
    pass 3 * keep_every when df is downsampled.

    Frozen ghosts (presence <= 0.2) and interpolated ball rows are excluded:
    style must be measured on real observations only.
    """
    players = df[(df["class"] != "ball") & (df.presence_weight > 0.2)]
    ball_all = df[(df["class"] == "ball") & (df.presence_weight >= 0.99)]
    ball = ball_all.drop_duplicates("frame_index").set_index("frame_index")

    acc = {t: {"line": [], "width": [], "press": [], "speeds": [],
               "nets": 0.0, "paths": 0.0, "n_poss": 0, "dirs": []}
           for t in (0, 1)}

    for _, pseg in players.groupby("segment_id"):
        dirs = {t: attack_dir(pseg, t) for t in (0, 1)}
        pg = {f: g for f, g in pseg.groupby("frame_index")}
        frames = sorted(pg)

        # per-frame possession (nearest player to ball, carry on loose ball)
        poss = {}
        last = None
        for f in frames:
            if f not in ball.index:
                poss[f] = last
                continue
            bx, by = ball.loc[f, "x_world"], ball.loc[f, "y_world"]
            g = pg[f]
            d = np.hypot(g.x_world.values - bx, g.y_world.values - by)
            j = d.argmin()
            last = int(g.team_id.values[j]) if d[j] <= POSS_R else last
            poss[f] = last

        for t in (0, 1):
            d = dirs[t]
            opp = 1 - t
            a = acc[t]
            a["dirs"].append(d)
            for f in frames:
                g = pg[f]
                of = g[(g.team_id == t) & (g.role_id == 1)]
                if len(of) >= MIN_VIS:
                    xf = _fwd(of.x_world.values, d)
                    if poss.get(f) == opp:                 # defending -> measure line
                        deep = np.sort(xf)[:DEEP_K]
                        a["line"].append(deep.mean() / PITCH_L)
                    a["width"].append((np.percentile(of.y_world.values, 90) -
                                       np.percentile(of.y_world.values, 10)) / PITCH_W)
                if poss.get(f) == opp and f in ball.index:
                    bx, by = ball.loc[f, "x_world"], ball.loc[f, "y_world"]
                    defs = g[g.team_id == t]
                    if len(defs):
                        dmin = np.hypot(defs.x_world.values - bx,
                                        defs.y_world.values - by).min()
                        a["press"].append(1.0 - np.clip(dmin / PRESS_NORM, 0, 1))

            # tempo / directness accumulation over this segment's possession runs
            poss_frames = [f for f in frames if poss.get(f) == t and f in ball.index]
            a["n_poss"] += len(poss_frames)
            if poss_frames:
                a["speeds"].extend(np.hypot(ball.loc[poss_frames, "vx"].values,
                                            ball.loc[poss_frames, "vy"].values).tolist())
                for r in _runs(poss_frames, max_gap):
                    if len(r) < 2:
                        continue
                    xs = _fwd(ball.loc[r, "x_world"].values, d)
                    ys = ball.loc[r, "y_world"].values
                    a["nets"] += xs[-1] - xs[0]
                    a["paths"] += np.hypot(np.diff(xs), np.diff(ys)).sum()

    out = {}
    for t in (0, 1):
        a = acc[t]
        tempo = (round(float(np.clip(np.median(a["speeds"]) / BALL_SPEED_NORM, 0, 1)), 3)
                 if len(a["speeds"]) >= 2 else None)
        direct = (round(float(np.clip(a["nets"] / a["paths"], 0, 1)), 3)
                  if a["n_poss"] >= 3 and a["paths"] > 1e-6 else None)
        td = None if (tempo is None or direct is None) else round(0.5 * tempo + 0.5 * direct, 3)
        width = _med(a["width"])
        out[f"team{t}"] = {
            "line_height": _med(a["line"]),
            "press": _med(a["press"]),
            "width": width,
            "compactness": round(1 - width, 3) if width is not None else None,
            "tempo": tempo,
            "directness": direct,
            "tempo_directness": td,
            "attack_dir": a["dirs"][0] if len(set(a["dirs"])) == 1 else a["dirs"],
            "n_line_frames": len(a["line"]),
            "n_press_frames": len(a["press"]),
            "n_width_frames": len(a["width"]),
        }
    return out


def _runs(frames, max_gap):
    runs, cur = [], [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if b - a <= max_gap:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    return runs


def _med(v):
    return round(float(np.median(v)), 3) if len(v) else None


def chunk_styles(df: pd.DataFrame, step_frames: int, max_gap: int = 3) -> list:
    """Split each segment into chunks of ~step_frames (frame-index units) and
    compute a style dict per chunk. The last chunk of a segment absorbs any
    short tail (< half a chunk). Returns [(segment_id, f_lo, f_hi, style), ...].

    Chunk-level styles give many diverse (style -> behaviour) samples per match
    for FiLM conditioning, instead of one washed-out average per game.
    """
    out = []
    for seg, g in df.groupby("segment_id"):
        f0, f1 = int(g.frame_index.min()), int(g.frame_index.max())
        bounds = list(range(f0, f1 + 1, step_frames))
        if len(bounds) > 1 and f1 - bounds[-1] < step_frames // 2:
            bounds.pop()                                  # tail absorbed by previous chunk
        for i, lo in enumerate(bounds):
            hi = (bounds[i + 1] - 1) if i + 1 < len(bounds) else f1
            sl = g[(g.frame_index >= lo) & (g.frame_index <= hi)]
            if sl.empty:
                continue
            out.append((int(seg), lo, hi, compute_style(sl, max_gap)))
    return out


def style_vector(team_style: dict) -> list:
    """The 4-knob vector used to condition the model; None -> 0.5 neutral."""
    keys = ["line_height", "press", "width", "tempo_directness"]
    return [0.5 if team_style.get(k) is None else team_style[k] for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gsr", default=str(ROOT / "output" / "gsr"))
    args = ap.parse_args()
    gsr = Path(args.gsr)
    clips = sorted(gsr.glob("SNGS-*/tracks.parquet"))
    summary = []
    for c in clips:
        df = pd.read_parquet(c)
        meta = json.loads((c.parent / "meta.json").read_text())
        st = compute_style(df)
        st["clip"] = c.parent.name
        st["action_class"] = meta.get("action_class")
        (c.parent / "style.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
        summary.append(st)
        t0 = st["team0"]
        print(f"{c.parent.name}: {meta.get('action_class'):14s} "
              f"line={t0['line_height']} press={t0['press']} "
              f"width={t0['width']} tempo={t0['tempo_directness']}")
    (gsr / "style_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    # distribution sanity
    for k in ["line_height", "press", "width", "tempo_directness"]:
        vals = [s[t][k] for s in summary for t in ("team0", "team1") if s[t][k] is not None]
        if vals:
            print(f"  {k:18s} n={len(vals)} "
                  f"min={min(vals):.2f} med={np.median(vals):.2f} max={max(vals):.2f}")


if __name__ == "__main__":
    main()
