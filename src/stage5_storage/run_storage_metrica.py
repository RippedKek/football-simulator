"""Stage 5 (Metrica variant) -- windowing for full-pitch tracking matches.

Differences from run_storage_gsr (which assumed 30 s single-segment clips):
  * iterates matches under output/metrica/, each with MULTIPLE segments
    (halves); windows never cross a segment boundary (teams swap ends there).
  * style knobs are computed per 60 s CHUNK via stage_style.chunk_styles, and
    each window carries its chunk's style -- many diverse conditioning samples
    per match instead of one average per game.
  * train/val split is the last `holdout_fraction` of each segment's timeline;
    windows straddling the boundary are dropped (stride < length would other-
    wise leak train frames into val).
  * skips the O(frames) possession flag (unused by windows; the style pass
    computes its own possession).

Output output/storage/metrica/windows_{train,val}.npz, same keys as GSR:
    agents (N,W,23,4)  presence (N,W,23)  target_delta (N,W,23,2)
    style0 (N,4)       style1 (N,4)

    python -m stage5_storage.run_storage_metrica
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage5_storage.common import (smooth_velocities, clamp_speed,
                                   segment_arrays, normalise)
from stage_style.descriptors import chunk_styles, style_vector

ROOT = Path(__file__).resolve().parents[2]


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def cap_deltas(td, nrm, target_fps):
    """Physics-cap per-step position targets: |delta| <= vmax * dt per agent
    type. Kills rare ball-teleport spikes (out-of-play interpolation jumps)
    that would otherwise dominate the squared trajectory loss."""
    dt = 1.0 / target_fps
    scale_m = np.array([nrm["pos_x_m"], nrm["pos_y_m"]], np.float32)
    n = np.linalg.norm(td * scale_m, axis=-1, keepdims=True)      # (W,23,1) metres/step
    cap = np.full((1, td.shape[1], 1), nrm["player_max_speed_mps"] * dt, np.float32)
    cap[0, 22, 0] = nrm["ball_max_speed_mps"] * dt
    s = np.minimum(1.0, cap / np.maximum(n, 1e-9))
    return (td * s).astype(np.float32)


def window_iter(seg_df, frames, cfg, styles):
    """Windows for one segment; each carries the style of the chunk containing
    its start frame. styles: [(f_lo, f_hi, sv0, sv1), ...]."""
    W, stride = cfg["window"]["length"], cfg["window"]["stride"]
    nrm = cfg["normalisation"]
    if len(frames) < W + 1:
        return
    a, pres = segment_arrays(seg_df, frames)
    a = normalise(a, nrm)
    pos = a[..., :2]
    for s in range(0, len(frames) - W, stride):
        f_start = frames[s]
        sv = next(((v0, v1) for lo, hi, v0, v1 in styles if lo <= f_start <= hi), None)
        if sv is None:
            continue
        e = s + W
        td = cap_deltas(pos[s + 1:e + 1] - pos[s:e], nrm, cfg["target_fps"])
        yield s, e, {
            "agents": a[s:e],
            "presence": pres[s:e],
            "target_delta": td,
            "style0": np.asarray(sv[0], np.float32),
            "style1": np.asarray(sv[1], np.float32),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=str(ROOT / "output" / "metrica"))
    ap.add_argument("--out", default=str(ROOT / "output" / "storage" / "metrica"))
    args = ap.parse_args()

    cfg = load_json(ROOT / "config" / "storage.json")
    sm = cfg["smoothing"]
    hold = cfg["validation"]["holdout_fraction"]
    chunk_secs = cfg.get("style_chunk_secs", 60)
    matches = sorted(Path(args.tracks).glob("*/tracks.parquet"))
    print(f"{len(matches)} matches, {chunk_secs}s style chunks, "
          f"val = last {hold:.0%} of each segment")

    train, val = [], []
    style_log = []
    for mp in matches:
        df = pd.read_parquet(mp)
        meta = load_json(mp.parent / "meta.json")
        fps = meta["fps"]
        keep_every = max(1, int(round(fps / cfg["target_fps"])))

        df = smooth_velocities(df, fps, sm["savgol_window"], sm["savgol_polyorder"])
        df = clamp_speed(df, cfg["normalisation"]["player_max_speed_mps"],
                         cfg["normalisation"]["ball_max_speed_mps"])

        n_tr = n_va = 0
        for seg in sorted(df.segment_id.unique()):
            seg_df = df[df.segment_id == seg]
            frames = sorted(seg_df.frame_index.unique().tolist())[::keep_every]
            ds = seg_df[seg_df.frame_index.isin(set(frames))].rename(columns={"class": "class_"})

            # per-chunk styles on the downsampled slice (style is a slow signal)
            chs = chunk_styles(ds.rename(columns={"class_": "class"}),
                               step_frames=int(chunk_secs * fps),
                               max_gap=3 * keep_every)
            styles = [(lo, hi, style_vector(st["team0"]), style_vector(st["team1"]))
                      for _, lo, hi, st in chs]
            style_log.extend({"match": mp.parent.name, "segment": int(seg),
                              "f_lo": lo, "f_hi": hi, "style": st}
                             for _, lo, hi, st in chs)

            vb = int(len(frames) * (1 - hold))            # val boundary (frame position)
            for s, e, w in window_iter(ds, frames, cfg, styles):
                if e <= vb:
                    train.append(w); n_tr += 1
                elif s >= vb:
                    val.append(w); n_va += 1              # straddlers dropped
        print(f"  {mp.parent.name}: train {n_tr} / val {n_va} windows")

    def stack(wins):
        keys = ("agents", "presence", "target_delta", "style0", "style1")
        if not wins:
            return {k: np.zeros((0,)) for k in keys}
        return {k: np.stack([w[k] for w in wins]) for k in keys}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va = stack(train), stack(val)
    np.savez_compressed(out_dir / "windows_train.npz", **tr)
    np.savez_compressed(out_dir / "windows_val.npz", **va)
    (out_dir / "style_chunks.json").write_text(json.dumps(style_log, indent=1),
                                               encoding="utf-8")

    sv = np.concatenate([tr["style0"], tr["style1"]]) if tr["style0"].size else np.zeros((0, 4))
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "metrica-full-pitch", "n_matches": len(matches),
        "target_fps": cfg["target_fps"], "window": cfg["window"],
        "normalisation": cfg["normalisation"], "n_agents": 23,
        "style_chunk_secs": chunk_secs,
        "train_windows": int(tr["agents"].shape[0]),
        "val_windows": int(va["agents"].shape[0]),
        "agents_order": "0-10 team0 (0=GK), 11-21 team1 (11=GK), 22 ball",
        "style_knobs": ["line_height", "press", "width", "tempo_directness"],
        "style_spread_p10_p90": [np.percentile(sv, 10, axis=0).round(3).tolist(),
                                 np.percentile(sv, 90, axis=0).round(3).tolist()]
        if sv.size else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"windows: train {meta['train_windows']} / val {meta['val_windows']} -> {out_dir}")
    if sv.size:
        print(f"style spread p10..p90 per knob: {meta['style_spread_p10_p90']}")


if __name__ == "__main__":
    main()
