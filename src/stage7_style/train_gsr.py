"""Train StyleNetGSR.

M3: teacher-forced. M4: scheduled-sampling rollout + physics priors -- the fix
for autoregressive corner-collapse.

Scheduled sampling (when rollout.enabled): each step a window's input positions
are, with probability p (annealed 0 -> p_max), replaced by the model's own
previous prediction instead of ground truth; features are rebuilt from those
positions with the differentiable Stage-6 port. The model thus learns to recover
from its own drift -- exactly the regime the simulator runs in. Physics priors
penalize leaving the pitch and super-human speed over the rolled positions.

    python -m stage7_style.train_gsr                       # M3 (rollout off in cfg)
    python -m stage7_style.train_gsr --rollout             # M4
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from stage7_style.model_gsr import StyleNetGSR, BALL
from stage7_style.features_torch import build_frame

ROOT = Path(__file__).resolve().parents[2]
N_SLOTS = 11


def structure_priors(nxt, mask, lc):
    """GK-anchor (keep GKs near a goal line) + anti-collapse spacing, on present agents.

    nxt: predicted next positions (B,T,23,2) normalised. mask: (B,T,23) present.
    """
    # GK anchor: penalise GK x far from the nearest goal line (x=0 or 1)
    gk = nxt[:, :, [0, N_SLOTS], 0]                       # (B,T,2)
    gkm = mask[:, :, [0, N_SLOTS]]
    dist_goal = torch.minimum(gk, 1 - gk)
    gk_pen = (F.relu(dist_goal - lc["gk_goal_dist"]) * gkm).sum() / (gkm.sum() + 1e-6)

    # spacing: nearest teammate distance (metres) per team's outfield
    scale = torch.tensor([105.0, 68.0], device=nxt.device)
    min_sep = lc["spacing_min_m"]
    sp = 0.0
    for idx in (slice(1, N_SLOTS), slice(N_SLOTS + 1, BALL)):
        P = nxt[:, :, idx, :] * scale                    # (B,T,10,2)
        M = mask[:, :, idx]                               # (B,T,10)
        d = torch.cdist(P, P)
        pm = M[..., None] * M[..., None, :]
        big = (1 - pm) * 1e3 + torch.eye(P.shape[2], device=nxt.device) * 1e3
        nearest = (d + big).min(-1).values               # (B,T,10)
        sp = sp + (F.relu(min_sep - nearest) * M).sum() / (M.sum() + 1e-6)
    return gk_pen, sp


def load_split(feat_dir, split):
    d = np.load(feat_dir / f"features_{split}.npz")
    if d["agent_feat"].size == 0:
        return None
    # also need raw normalised agents (positions) for rollout re-featurization
    return d


def make_dataset(feat, store, split):
    d = load_split(feat, split)
    if d is None:
        return None
    ag = np.load(store / f"windows_{split}.npz")["agents"]   # (N,T,23,4) normalised
    t = lambda x: torch.from_numpy(x).float()
    return TensorDataset(t(d["agent_feat"]), t(d["presence"]), t(d["target_delta"]),
                         t(d["style0"]), t(d["style1"]), t(ag))


def vel_factor(norm, fps, device):
    """normalised-position delta per frame -> normalised-velocity units, per agent."""
    L, Wd = norm["pos_x_m"], norm["pos_y_m"]
    pmax, bmax = norm["player_max_speed_mps"], norm["ball_max_speed_mps"]
    fac = torch.zeros(23, 2, device=device)
    fac[:BALL, 0] = L * fps / pmax; fac[:BALL, 1] = Wd * fps / pmax
    fac[BALL, 0] = L * fps / bmax; fac[BALL, 1] = Wd * fps / bmax
    return fac


def make_agents(samp, fac):
    """samp (B,T,23,2) normalised pos -> agents (B,T,23,4) with consistent velocity."""
    vel = torch.zeros_like(samp)
    vel[:, 1:] = (samp[:, 1:] - samp[:, :-1]) * fac
    return torch.cat([samp, vel], dim=-1)


def traj_loss(delta, target, presence, mask_thr, ball_w):
    w = (presence > mask_thr).float()
    w = w.clone(); w[..., BALL] *= ball_w
    se = ((delta - target) ** 2).sum(-1)
    return (w * se).sum() / (w.sum() + 1e-6)


def forward_losses(model, batch, cfg, dev, p, fac):
    feat, pres, tgt, s0, s1, agents = [b.to(dev) for b in batch]
    if cfg.get("no_style"):                     # unconditioned: pure "how football moves"
        s0 = torch.full_like(s0, 0.5)
        s1 = torch.full_like(s1, 0.5)
    lc = cfg["loss"]
    mt = lc["presence_mask_threshold"]
    B, T = feat.shape[0], feat.shape[1]

    if p > 0:
        with torch.no_grad():
            d0, _, _ = model(feat, pres, s0, s1, mt)
        pos = agents[..., :2]
        samp = pos.clone()
        use = (torch.rand(B, T, device=dev) < p)
        for t in range(1, T):
            pred = samp[:, t - 1] + d0[:, t - 1]
            samp[:, t] = torch.where(use[:, t][:, None, None], pred, pos[:, t])
        ag_s = make_agents(samp, fac)
        feat = build_frame(ag_s.reshape(B * T, 23, 4), cfg["norm"]).reshape(B, T, 23, 24)

    delta, style_pred, event = model(feat, pres, s0, s1, mt)
    lt = traj_loss(delta, tgt, pres, mt, lc["ball_weight"])
    ls = F.mse_loss(style_pred, torch.stack([s0, s1], 1))
    le = F.cross_entropy(event, torch.zeros(B, dtype=torch.long, device=dev))

    # physics + structure priors over predicted next positions
    nxt = agents[..., :2] + delta
    mask = (pres > mt).float()
    oob = (F.relu(nxt - 1.0) + F.relu(-nxt)).mean()
    speed = (delta * fac).norm(dim=-1)                  # normalised speed proxy
    spd = F.relu(speed - 1.0).mean()
    gk_pen, sp = structure_priors(nxt, mask, lc)
    rl = cfg["train"]["rollout"]
    loss = (lc["traj_weight"] * lt + lc["style_weight"] * ls + lc["event_weight"] * le
            + rl["oob_weight"] * oob + rl["speed_weight"] * spd
            + lc["gk_anchor_weight"] * gk_pen + lc["spacing_weight"] * sp)
    return loss, {"traj": lt.item(), "style": ls.item(), "event": le.item(),
                  "oob": float(oob.item()), "spd": float(spd.item()),
                  "gk": float(gk_pen.item()), "sp": float(sp.item())}


def run_epoch(model, loader, opt, scaler, cfg, dev, train, p, fac):
    model.train(train)
    agg = {k: 0.0 for k in ("traj", "style", "event", "oob", "spd", "gk", "sp")}
    n = 0
    for batch in loader:
        with torch.set_grad_enabled(train):
            loss, parts = forward_losses(model, batch, cfg, dev, p if train else 0.0, fac)
        if train:
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            scaler.step(opt); scaler.update()
        b = batch[0].shape[0]; n += b
        for k in agg:
            agg[k] += parts[k] * b
    return {k: agg[k] / max(n, 1) for k in agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=str(ROOT / "output" / "features" / "gsr"))
    ap.add_argument("--store", default=str(ROOT / "output" / "storage" / "gsr"))
    ap.add_argument("--out", default=str(ROOT / "output" / "training" / "gsr"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--rollout", action="store_true", help="enable scheduled sampling (M4)")
    ap.add_argument("--no-style", action="store_true",
                    help="train unconditioned (neutral style) -- pure football dynamics")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config" / "training_gsr.json").read_text())
    store = Path(args.store)
    smeta = json.loads((store / "meta.json").read_text())
    cfg["norm"] = smeta["normalisation"]
    fps = smeta["target_fps"]
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.rollout:
        cfg["train"]["rollout"]["enabled"] = True
    cfg["no_style"] = args.no_style
    rl = cfg["train"]["rollout"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fac = vel_factor(cfg["norm"], fps, dev)

    tr = make_dataset(Path(args.feat), store, "train")
    va = make_dataset(Path(args.feat), store, "val")
    bs = cfg["train"]["batch_size"]
    tr_loader = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    va_loader = DataLoader(va, batch_size=bs) if va else None
    print(f"device {dev}  train {len(tr)}  val {len(va) if va else 0}  rollout {rl['enabled']}")

    model = StyleNetGSR(cfg).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    o = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], betas=tuple(o["betas"]),
                            weight_decay=o["weight_decay"])
    E = cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    log = []
    for ep in range(E):
        # scheduled-sampling probability: 0 until start, then ramp to p_max
        p = 0.0
        if rl["enabled"] and ep >= rl["start_epoch"]:
            frac = (ep - rl["start_epoch"]) / max(1, E - 1 - rl["start_epoch"])
            p = rl["p_max"] * min(1.0, frac)
        trm = run_epoch(model, tr_loader, opt, scaler, cfg, dev, True, p, fac)
        sched.step()
        rec = {"epoch": ep, "p": round(p, 3), "train": trm}
        if va_loader and (ep % cfg["train"]["val_every"] == 0 or ep == E - 1):
            with torch.no_grad():
                rec["val"] = run_epoch(model, va_loader, opt, scaler, cfg, dev, False, 0.0, fac)
        log.append(rec)
        msg = (f"ep {ep:3d} p{p:.2f} traj {trm['traj']:.4f} gk {trm['gk']:.3f} "
               f"sp {trm['sp']:.3f} oob {trm['oob']:.4f}")
        if "val" in rec:
            msg += f"  | val traj {rec['val']['traj']:.4f}"
        print(msg, flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "model": model.state_dict()}, out_dir / "style_model.pt")
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": dev, "epochs": E, "rollout": rl, "final": log[-1]}, indent=2), encoding="utf-8")
    print(f"saved -> {out_dir/'style_model.pt'}")


if __name__ == "__main__":
    main()
