"""Train KickNet on the Metrica kick-decision dataset.

Class-weighted action CE (47 shots vs 2243 passes) + receiver CE on labeled
completed passes. 15% random holdout.

    python -m stage7_style.train_kick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stage7_style.kick_model import KickNet

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "output" / "kick"))
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    d = np.load(Path(args.data) / "kick_data.npz")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    N = d["ctx"].shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    cut = int(N * 0.85)
    tr_i, va_i = idx[:cut], idx[cut:]

    t = lambda x: torch.from_numpy(x).to(dev)
    ctx, mates = t(d["ctx"]).float(), t(d["mates"]).float()
    mask, act, recv = t(d["mask"]).float(), t(d["action"]), t(d["receiver"])

    counts = np.bincount(d["action"], minlength=3)
    w = torch.tensor(N / (3.0 * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    print(f"n={N} hold/pass/shot={counts.tolist()} class-w={w.cpu().numpy().round(2).tolist()}")

    model = KickNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def split_loss(ix, train):
        model.train(train)
        with torch.set_grad_enabled(train):
            a, r = model(ctx[ix], mates[ix], mask[ix])
            la = F.cross_entropy(a, act[ix], weight=w)
            has = recv[ix] >= 0
            lr_ = (F.cross_entropy(r[has], recv[ix][has]) if has.any()
                   else torch.zeros((), device=dev))
            return la + lr_, a, r, has

    for ep in range(args.epochs):
        perm = torch.randperm(len(tr_i), device=dev)
        for b in range(0, len(tr_i), 512):
            ix = t(tr_i)[perm[b:b + 512]]
            loss, *_ = split_loss(ix, True)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    # ---- final metrics ----
    def metrics(ix_np, name):
        ix = t(ix_np)
        with torch.no_grad():
            _, a, r, has = split_loss(ix, False)
        pred = a.argmax(1)
        acc = (pred == act[ix]).float().mean().item()
        rec = {}
        for c, nm in enumerate(("hold", "pass", "shot")):
            m = act[ix] == c
            rec[nm] = (pred[m] == c).float().mean().item() if m.any() else float("nan")
        top1 = top3 = float("nan")
        if has.any():
            rl, rt = r[has], recv[ix][has]
            top1 = (rl.argmax(1) == rt).float().mean().item()
            top3 = (rl.topk(3, dim=1).indices == rt[:, None]).any(1).float().mean().item()
        print(f"{name}: action acc {acc:.3f} recall h/p/s "
              f"{rec['hold']:.2f}/{rec['pass']:.2f}/{rec['shot']:.2f} "
              f"receiver top1 {top1:.3f} top3 {top3:.3f}")
        return {"acc": acc, "recall": rec, "recv_top1": top1, "recv_top3": top3}

    mtr = metrics(tr_i, "train")
    mva = metrics(va_i, "val")

    out = Path(args.data) / "kick_model.pt"
    torch.save({"model": model.state_dict(), "val": mva, "train": mtr}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
