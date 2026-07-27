"""Stage 0 -- Metrica Sports open tracking-data loader.

Full-pitch positional data: ALL 22 players + ball, every frame, whole matches.
This is the data type the broadcast-GSR experiments proved necessary (docs/10):
no partial visibility, no ball-centred crop, so the model can finally learn
global formation structure.

Three sample matches (github.com/metrica-sports/sample-data):
  game1, game2  CSV pairs (Home/Away), normalised 0-1 coords, 25 fps
  game3         FIFA-EPTS txt + metadata xml (channel order changes per
                substitution -- each DataFormatSpecification covers a frame range)

Output: output/metrica/<game>/tracks.parquet in the exact Stage 5 schema
(frame_index, timestamp, segment_id, team_id, slot_id, role_id, class, x_world,
y_world, vx, vy, confidence, presence_weight, source) + meta.json.

Conventions:
  * segment_id = half (0/1). Teams swap ends at half-time and Metrica does NOT
    flip coords, so nothing downstream may span a segment boundary.
  * fixed 11 slots/team: slot 0 = GK, subs inherit the slot their predecessor
    vacated (interval scheduling), so a slot is one "shirt" for the whole match.
  * gaps in a slot (between sub-off and sub-on, red cards, tracking blips) are
    freeze-filled at the last known position with presence 0.1: masked from loss
    and attention (thr 0.2) but keeps relational features (nearest-opp, centroid)
    away from the (0,0) corner that absent-agent zeros used to inject.
  * ball out-of-play gaps are time-interpolated: presence 0.5 for short gaps
    (<= 2 s), 0.2 for longer; never NaN, never (0,0).

    python -m stage0_gsr.load_metrica --all
"""

from __future__ import annotations

import argparse
import heapq
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PITCH_L, PITCH_W = 105.0, 68.0
N_SLOTS = 11
PLAYER_V_MAX = 12.0     # m/s clamp for players
BALL_V_MAX = 40.0       # m/s clamp for the ball (shots ~30+; old GSR loader wrongly capped at 12)
GK_XDIST = 0.35         # mean |x-0.5| above this (normalised) = goalkeeper (CSV files)
BALL_SHORT_GAP_S = 2.0


# --------------------------------------------------------------------------- #
# CSV games (1, 2)
# --------------------------------------------------------------------------- #
def read_team_csv(path: Path, team_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One Metrica tracking CSV -> (long player df, ball df). Coords stay 0-1."""
    with open(path, "r", encoding="utf-8") as f:
        f.readline()                                   # row 1: team name markers
        jerseys = f.readline().rstrip("\n").split(",")  # row 2: jersey numbers
        names = f.readline().rstrip("\n").split(",")    # row 3: column names

    cols = ["period", "frame", "time"]
    players = []
    i = 3
    while i < len(names):
        if names[i].startswith("Player"):
            j = jerseys[i]
            cols += [f"p{j}_x", f"p{j}_y"]
            players.append(j)
            i += 2
        elif names[i] == "Ball":
            cols += ["ball_x", "ball_y"]
            i += 2
        else:
            i += 1
    df = pd.read_csv(path, skiprows=3, names=cols)

    long = []
    for j in players:
        sub = pd.DataFrame({
            "frame_index": df["frame"].astype(int),
            "period": df["period"].astype(int),
            "x": df[f"p{j}_x"], "y": df[f"p{j}_y"],
        })
        sub["player_key"] = f"t{team_id}_j{j}"
        long.append(sub.dropna(subset=["x", "y"]))
    pl = pd.concat(long, ignore_index=True)
    pl["team_id"] = team_id

    ball = df[["frame", "period", "ball_x", "ball_y"]].rename(
        columns={"frame": "frame_index", "ball_x": "x", "ball_y": "y"})
    ball["frame_index"] = ball["frame_index"].astype(int)
    return pl, ball


def load_csv_game(game_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    stem = game_dir.name                                # Sample_Game_1
    home = game_dir / f"{stem}_RawTrackingData_Home_Team.csv"
    away = game_dir / f"{stem}_RawTrackingData_Away_Team.csv"
    h_pl, h_ball = read_team_csv(home, 0)
    a_pl, _ = read_team_csv(away, 1)
    players = pd.concat([h_pl, a_pl], ignore_index=True)

    # GK = player who lives near a goal line. Per-row |x-0.5| (NOT |mean-0.5|:
    # ends swap at half-time, so a GK's mean x is ~0.5 over the match).
    roles = {}
    for pk, g in players.groupby("player_key"):
        roles[pk] = "goalkeeper" if g.x.sub(0.5).abs().mean() > GK_XDIST else "player"
    fps = 25.0
    return players, h_ball, roles, fps


# --------------------------------------------------------------------------- #
# EPTS game (3)
# --------------------------------------------------------------------------- #
def load_epts_game(game_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    stem = game_dir.name
    meta = ET.parse(game_dir / f"{stem}_metadata.xml").getroot()

    fps = float(meta.findtext(".//FrameRate"))
    halves = {p.findtext("Name"): p.findtext("Value")
              for p in meta.findall(".//GlobalConfig//ProviderParameter")}
    h1_end = int(halves["first_half_end"])

    team_ids = {}
    for i, t in enumerate(meta.findall(".//Team")):
        team_ids[t.get("id")] = i                      # FIFATMA -> 0, FIFATMB -> 1
    pinfo, roles = {}, {}
    for p in meta.findall(".//Player"):
        pid = p.get("id")
        pos = None
        for pp in p.findall(".//ProviderParameter"):
            if pp.findtext("Name") == "position_type":
                pos = pp.findtext("Value")
        pinfo[pid] = team_ids[p.get("teamId")]
        roles[pid] = "goalkeeper" if pos == "Goalkeeper" else "player"

    chan_player = {c.get("id"): c.get("playerId")
                   for c in meta.findall(".//PlayerChannel")}
    specs = []                                          # (start, end, [playerId,...])
    for spec in meta.findall(".//DataFormatSpecification"):
        order = [chan_player[r.get("playerChannelId")]
                 for r in spec.findall(".//PlayerChannelRef")
                 if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    specs.sort()

    # parse txt per spec block (fixed column count inside a block -> vectorized)
    lines = (game_dir / f"{stem}_tracking.txt").read_text().splitlines()
    si, buf, blocks = 0, [], []
    for ln in lines:
        frame = int(ln.split(":", 1)[0])
        while frame > specs[si][1]:
            if buf:
                blocks.append((specs[si][2], buf)); buf = []
            si += 1
        buf.append(ln)
    if buf:
        blocks.append((specs[si][2], buf))

    prows, brows = [], []
    for order, blk in blocks:
        txt = "\n".join(l.replace(":", ",").replace(";", ",") for l in blk)
        arr = np.genfromtxt(io.StringIO(txt), delimiter=",")
        if arr.ndim == 1:
            arr = arr[None]
        frames = arr[:, 0].astype(int)
        periods = np.where(frames <= h1_end, 1, 2)
        for k, pid in enumerate(order):
            x, y = arr[:, 1 + 2 * k], arr[:, 2 + 2 * k]
            ok = ~(np.isnan(x) | np.isnan(y))
            if not ok.any():
                continue
            prows.append(pd.DataFrame({
                "frame_index": frames[ok], "period": periods[ok],
                "x": x[ok], "y": y[ok],
                "player_key": pid, "team_id": pinfo[pid]}))
        brows.append(pd.DataFrame({
            "frame_index": frames, "period": periods,
            "x": arr[:, -2], "y": arr[:, -1]}))

    players = pd.concat(prows, ignore_index=True)
    ball = pd.concat(brows, ignore_index=True).sort_values("frame_index")
    return players, ball, roles, fps


# --------------------------------------------------------------------------- #
# shared post-processing
# --------------------------------------------------------------------------- #
def assign_slots(players: pd.DataFrame, roles: dict) -> dict:
    """player_key -> slot. GKs share slot 0; outfield slots by interval scheduling
    (a sub takes the slot its predecessor vacated)."""
    iv = players.groupby("player_key").frame_index.agg(["min", "max"])
    slots = {}
    for tid in (0, 1):
        keys = [k for k in iv.index if players[players.player_key == k].team_id.iloc[0] == tid]
        gks = sorted([k for k in keys if roles[k] == "goalkeeper"], key=lambda k: iv.loc[k, "min"])
        for k in gks:
            slots[k] = 0
        outf = sorted([k for k in keys if roles[k] != "goalkeeper"], key=lambda k: iv.loc[k, "min"])
        free = list(range(1, N_SLOTS))
        ends = []                                       # heap (last_frame, slot)
        for k in outf:
            f0, f1 = iv.loc[k, "min"], iv.loc[k, "max"]
            while ends and ends[0][0] <= f0:
                _, s = heapq.heappop(ends)
                free.append(s); free.sort()
            if free:
                s = free.pop(0)
            else:                                       # data-overlap anomaly: steal soonest-ending
                _, s = heapq.heappop(ends)
            slots[k] = s
            heapq.heappush(ends, (f1, s))
    return slots


def densify_players(players: pd.DataFrame, roles: dict, slots: dict, fps: float) -> pd.DataFrame:
    """Per (team, slot, segment): full frame coverage. Real rows presence 1.0,
    freeze-filled gaps presence 0.1 (masked but positioned). Velocity from real
    positions, 0 on frozen rows, clamped."""
    players = players.copy()
    players["slot_id"] = players.player_key.map(slots)
    players["role"] = players.player_key.map(roles)
    players["segment_id"] = players.period - 1
    players = players.sort_values(["team_id", "slot_id", "frame_index"])
    players = players.drop_duplicates(["team_id", "slot_id", "frame_index"], keep="first")

    seg_frames = {s: np.sort(players[players.segment_id == s].frame_index.unique())
                  for s in players.segment_id.unique()}
    out = []
    for (tid, sid, seg), g in players.groupby(["team_id", "slot_id", "segment_id"]):
        frames = seg_frames[seg]
        g = g.set_index("frame_index").reindex(frames)
        real = g.x.notna().values
        if not real.any():
            continue
        g["x"] = g.x.ffill().bfill()
        g["y"] = g.y.ffill().bfill()
        dt = np.diff(frames, prepend=frames[0]) / fps
        dt[0] = 1.0
        vx = np.diff(g.x.values, prepend=g.x.values[0]) / dt
        vy = np.diff(g.y.values, prepend=g.y.values[0]) / dt
        vx[~real] = 0.0
        vy[~real] = 0.0
        d = pd.DataFrame({
            "frame_index": frames, "segment_id": seg, "team_id": tid, "slot_id": sid,
            "x": g.x.values, "y": g.y.values, "vx_n": vx, "vy_n": vy,
            "presence_weight": np.where(real, 1.0, 0.1),
            "source": np.where(real, "metrica", "freeze"),
        })
        d["role_id"] = 0 if sid == 0 else 1
        d["class"] = "goalkeeper" if sid == 0 else "player"
        out.append(d)
    return pd.concat(out, ignore_index=True)


def densify_ball(ball: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Interpolate out-of-play gaps; presence 1 real / 0.5 short gap / 0.2 long."""
    b = ball.drop_duplicates("frame_index").sort_values("frame_index").reset_index(drop=True)
    real = b.x.notna().values
    b["x"] = b.x.interpolate(limit_direction="both")
    b["y"] = b.y.interpolate(limit_direction="both")

    pres = np.ones(len(b), np.float32)
    gap_id = (real != np.roll(real, 1)).cumsum()
    for gid in np.unique(gap_id[~real]):
        m = (gap_id == gid) & ~real
        pres[m] = 0.5 if m.sum() <= BALL_SHORT_GAP_S * fps else 0.2
    pres[real] = 1.0

    frames = b.frame_index.values
    dt = np.diff(frames, prepend=frames[0]) / fps
    dt[0] = 1.0
    vx = np.diff(b.x.values, prepend=b.x.values[0]) / dt
    vy = np.diff(b.y.values, prepend=b.y.values[0]) / dt
    return pd.DataFrame({
        "frame_index": frames, "segment_id": b.period.values - 1, "team_id": -1,
        "slot_id": 0, "role_id": 1, "class": "ball",
        "x": b.x.values, "y": b.y.values, "vx_n": vx, "vy_n": vy,
        "presence_weight": pres,
        "source": np.where(real, "metrica", "ball_interp"),
    })


def finalise(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """0-1 coords -> world metres, velocity to m/s + clamp, schema columns."""
    df = df.copy()
    df["x_world"] = np.clip(df.x * PITCH_L, 0, PITCH_L)
    df["y_world"] = np.clip(df.y * PITCH_W, 0, PITCH_W)
    df["vx"] = df.vx_n * PITCH_L
    df["vy"] = df.vy_n * PITCH_W
    sp = np.hypot(df.vx, df.vy)
    cap = np.where(df["class"] == "ball", BALL_V_MAX, PLAYER_V_MAX)
    scale = np.where(sp > cap, cap / np.maximum(sp, 1e-6), 1.0)
    df["vx"] *= scale
    df["vy"] *= scale
    df["timestamp"] = df.frame_index / fps
    df["confidence"] = 1.0
    cols = ["frame_index", "timestamp", "segment_id", "team_id", "slot_id", "role_id",
            "class", "x_world", "y_world", "vx", "vy", "confidence", "presence_weight", "source"]
    return df[cols].sort_values(["frame_index", "team_id", "slot_id"]).reset_index(drop=True)


def process_game(game_dir: Path, out_dir: Path, tag: str) -> dict:
    if (game_dir / f"{game_dir.name}_metadata.xml").exists():
        players, ball, roles, fps = load_epts_game(game_dir)
    else:
        players, ball, roles, fps = load_csv_game(game_dir)

    slots = assign_slots(players, roles)
    dense = densify_players(players, roles, slots, fps)
    bdf = densify_ball(ball, fps)
    df = finalise(pd.concat([dense, bdf], ignore_index=True), fps)

    out = out_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "tracks.parquet", index=False)

    # player_key -> (team_id, slot_id): needed by the kick-model builder to map
    # event-CSV player names onto agent slots
    team_of = players.groupby("player_key").team_id.first()
    slot_map = {pk: {"team_id": int(team_of[pk]), "slot_id": int(s), "role": roles[pk]}
                for pk, s in slots.items()}
    (out / "slot_map.json").write_text(json.dumps(slot_map, indent=1), encoding="utf-8")

    n_frames = df.frame_index.nunique()
    per_frame = df[(df["class"] != "ball") & (df.presence_weight > 0.2)] \
        .groupby(["frame_index", "team_id"]).size()
    speed = np.hypot(df.vx, df.vy)
    pball = df[df["class"] == "ball"]
    n_players = {t: len({k for k, s in slots.items()
                         if players[players.player_key == k].team_id.iloc[0] == t})
                 for t in (0, 1)}
    meta = {
        "game": tag, "fps": fps, "n_frames": int(n_frames),
        "rows": int(len(df)),
        "segments": sorted(int(s) for s in df.segment_id.unique()),
        "players_used": n_players,
        "visibility_mean": round(float(per_frame.mean()), 2),
        "visibility_min": int(per_frame.min()),
        "ball_real_frac": round(float((pball.source == "metrica").mean()), 3),
        "freeze_frac": round(float((df.source == "freeze").mean()), 4),
        "speed_p50": round(float(np.percentile(speed, 50)), 2),
        "speed_p99": round(float(np.percentile(speed, 99)), 2),
        "speed_max": round(float(speed.max()), 2),
        "duration_min": round(n_frames / fps / 60, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{tag}: {meta['duration_min']}min vis={meta['visibility_mean']} "
          f"ball_real={meta['ball_real_frac']*100:.0f}% freeze={meta['freeze_frac']*100:.2f}% "
          f"vmax={meta['speed_max']} rows={meta['rows']}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "metrica" / "data"))
    ap.add_argument("--out", default=str(ROOT / "output" / "metrica"))
    ap.add_argument("--game", help="single game dir name, e.g. Sample_Game_1")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    games = ([data / args.game] if args.game
             else sorted(p for p in data.iterdir() if p.is_dir()))
    metas = []
    for g in games:
        tag = f"game{g.name[-1]}" if g.name[-1].isdigit() else g.name
        metas.append(process_game(g, out, tag))
    (out / "summary.json").write_text(json.dumps(metas, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
