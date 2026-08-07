"""Hybrid engine M1 -- kick-decision dataset from Metrica event CSVs.

Every PASS / SHOT / BALL LOST event is a "kick decision" taken by the player in
possession; frames shortly before a pass (while the same player still carries
the ball) are "hold" negatives. Features are computed in the KICKER'S attack
frame (kicker always attacks +x, goal at (105, 34)), so one model serves both
teams and both halves.

Per sample:
  ctx (8):    kicker x/105, y/68, goal dist/105, cos+sin goal angle,
              nearest-opp dist/20, is_gk, opp-within-10m count /11
  mates (11,6): per teammate slot: dx/50, dy/34, dist/50,
              teammate's nearest-opp dist/20, teammate goal dist/105, is_gk
  mask (11):  1 = valid receiver (present, not the kicker)
  action:     0 hold, 1 pass (incl. BALL LOST = failed attempt), 2 shot
  receiver:   team-local slot 0-10 (completed passes only, else -1)

Also fits pass speed vs distance (for ballistic flight in the sim) and writes
base action rates -> output/kick/kick_meta.json.

    python -m stage7_style.build_kick_data
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PITCH_L, PITCH_W = 105.0, 68.0
GOAL = np.array([PITCH_L, PITCH_W / 2])
N_SLOTS = 11
FPS = 25.0
HOLD_BACK_FRAMES = (10, 25, 40)     # candidate offsets before a pass for hold samples
CARRY_R = 2.5                       # m: kicker must be this close to ball to count as carrying


def attack_dirs(tracks: pd.DataFrame) -> dict:
    """(team_id, segment_id) -> +1 attacks +x, -1 attacks -x (from GK mean x)."""
    out = {}
    gk = tracks[(tracks.slot_id == 0) & (tracks["class"] == "goalkeeper")]
    for (t, s), g in gk.groupby(["team_id", "segment_id"]):
        out[(t, s)] = 1.0 if g.x_world.mean() < PITCH_L / 2 else -1.0
    return out


def fwd(xy, d):
    """World coords -> attack frame of a team with direction d."""
    xy = np.asarray(xy, float).copy()
    if d < 0:
        xy[..., 0] = PITCH_L - xy[..., 0]
        xy[..., 1] = PITCH_W - xy[..., 1]
    return xy


def frame_state(fr: dict, f: int):
    return fr.get(f)


def build_sample(g: pd.DataFrame, kicker_team: int, kicker_slot: int, d: float):
    """g = tracks rows of one frame. Returns (ctx, mates, mask) or None."""
    pl = g[(g["class"] != "ball") & (g.presence_weight > 0.2)]
    own = pl[pl.team_id == kicker_team]
    opp = pl[pl.team_id != kicker_team]
    krow = own[own.slot_id == kicker_slot]
    if krow.empty or opp.empty:
        return None
    k = fwd(krow[["x_world", "y_world"]].values[0], d)
    opp_xy = fwd(opp[["x_world", "y_world"]].values, d)

    vg = GOAL - k
    dist_goal = np.linalg.norm(vg)
    d_opp = np.linalg.norm(opp_xy - k, axis=1)
    ctx = np.array([
        k[0] / PITCH_L, k[1] / PITCH_W, dist_goal / PITCH_L,
        vg[0] / max(dist_goal, 1e-6), vg[1] / max(dist_goal, 1e-6),
        min(d_opp.min(), 20) / 20, 1.0 if kicker_slot == 0 else 0.0,
        (d_opp <= 10).sum() / 11.0,
    ], np.float32)

    mates = np.zeros((N_SLOTS, 6), np.float32)
    mask = np.zeros(N_SLOTS, np.float32)
    for r in own.itertuples():
        s = int(r.slot_id)
        if s == kicker_slot:
            continue
        m = fwd((r.x_world, r.y_world), d)
        rel = m - k
        od = np.linalg.norm(opp_xy - m, axis=1).min()
        mates[s] = [rel[0] / 50, rel[1] / 34, np.linalg.norm(rel) / 50,
                    min(od, 20) / 20, np.linalg.norm(GOAL - m) / PITCH_L,
                    1.0 if s == 0 else 0.0]
        mask[s] = 1.0
    return ctx, mates, mask


def process_game(game_tag: str, data_dir: Path, out_rows: dict):
    gdir = ROOT / "output" / "metrica" / game_tag
    tracks = pd.read_parquet(gdir / "tracks.parquet")
    slot_map = json.loads((gdir / "slot_map.json").read_text())
    dirs = attack_dirs(tracks)

    n = int(game_tag[-1])
    ev = pd.read_csv(data_dir / f"Sample_Game_{n}" / f"Sample_Game_{n}_RawEventsData.csv")
    ev.columns = [c.strip() for c in ev.columns]
    team_id = {"Home": 0, "Away": 1}

    def player_slot(name, tid):
        m = re.match(r"Player(\d+)", str(name))
        if not m:
            return None
        rec = slot_map.get(f"t{tid}_j{m.group(1)}")
        return None if rec is None or rec["team_id"] != tid else rec["slot_id"]

    kicks = ev[ev.Type.isin(["PASS", "SHOT", "BALL LOST"])].copy()
    need = set(kicks["Start Frame"].astype(int))
    for f0 in kicks["Start Frame"].astype(int):
        need.update(f0 - k for k in HOLD_BACK_FRAMES)
    sub = tracks[tracks.frame_index.isin(need)]
    fr = {f: g for f, g in sub.groupby("frame_index")}
    ball = {f: g[g["class"] == "ball"] for f, g in fr.items()}

    speed_pts = []
    rng = np.random.default_rng(0)
    for r in kicks.itertuples():
        tid = team_id.get(r.Team)
        if tid is None:
            continue
        f0 = int(r._5)                                    # Start Frame
        seg = int(r.Period) - 1
        dirn = dirs.get((tid, seg))
        kslot = player_slot(r.From, tid)
        g = frame_state(fr, f0)
        if dirn is None or kslot is None or g is None:
            continue
        s = build_sample(g, tid, kslot, dirn)
        if s is None:
            continue
        ctx, mates, mask = s

        if r.Type == "SHOT":
            action, recv = 2, -1
        else:
            action = 1
            recv = player_slot(r.To, tid) if r.Type == "PASS" else None
            recv = -1 if recv is None or recv == kslot or mask[recv] == 0 else recv
            # pass speed sample (completed passes with sane coords/time)
            f1 = int(r._7)                                # End Frame
            if (r.Type == "PASS" and f1 > f0
                    and np.isfinite([r._11, r._12, r._13, r._14]).all()):
                p0 = np.array([r._11 * PITCH_L, r._12 * PITCH_W])
                p1 = np.array([r._13 * PITCH_L, r._14 * PITCH_W])
                dist = np.linalg.norm(p1 - p0)
                dt = (f1 - f0) / FPS
                if 1 < dist < 70 and 0.1 < dt < 5:
                    speed_pts.append((dist, dist / dt))
        out_rows["ctx"].append(ctx); out_rows["mates"].append(mates)
        out_rows["mask"].append(mask); out_rows["action"].append(action)
        out_rows["receiver"].append(recv)

        # hold negative: an earlier frame of the same carry (kicker near ball)
        if r.Type == "PASS":
            for k in rng.permutation(HOLD_BACK_FRAMES):
                fh = f0 - int(k)
                gh, bh = frame_state(fr, fh), ball.get(fh)
                if gh is None or bh is None or bh.empty or bh.presence_weight.iloc[0] < 0.99:
                    continue
                kr = gh[(gh.team_id == tid) & (gh.slot_id == kslot) & (gh["class"] != "ball")]
                if kr.empty:
                    continue
                bd = np.hypot(kr.x_world.iloc[0] - bh.x_world.iloc[0],
                              kr.y_world.iloc[0] - bh.y_world.iloc[0])
                if bd > CARRY_R:
                    continue
                sh = build_sample(gh, tid, kslot, dirn)
                if sh is not None:
                    out_rows["ctx"].append(sh[0]); out_rows["mates"].append(sh[1])
                    out_rows["mask"].append(sh[2]); out_rows["action"].append(0)
                    out_rows["receiver"].append(-1)
                break
    return speed_pts, len(kicks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "metrica" / "data"))
    ap.add_argument("--out", default=str(ROOT / "output" / "kick"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {k: [] for k in ("ctx", "mates", "mask", "action", "receiver")}
    speed_pts = []
    for tag in ("game1", "game2"):                        # game3 events are a different format
        sp, nk = process_game(tag, Path(args.data), rows)
        speed_pts.extend(sp)
        print(f"{tag}: {nk} kick events")

    d = {k: np.asarray(v, np.float32 if k in ("ctx", "mates", "mask") else np.int64)
         for k, v in rows.items()}
    np.savez_compressed(out_dir / "kick_data.npz", **d)

    sp = np.array(speed_pts)
    a, b = np.polyfit(sp[:, 0], sp[:, 1], 1)[::-1]        # speed ~= a + b*dist
    act = d["action"]
    meta = {
        "n_samples": int(len(act)),
        "n_hold": int((act == 0).sum()), "n_pass": int((act == 1).sum()),
        "n_shot": int((act == 2).sum()),
        "n_receiver_labeled": int((d["receiver"] >= 0).sum()),
        "pass_speed_fit": {"intercept": round(float(a), 3), "slope": round(float(b), 4),
                           "clip": [6.0, 30.0]},
        "pass_speed_med": round(float(np.median(sp[:, 1])), 2),
    }
    (out_dir / "kick_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
