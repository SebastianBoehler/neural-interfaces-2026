"""Multi-corpus supervised pretraining of the deep families (R10).

Usage: python experiments/pretrain.py ARCH ALIGN TAG CORPUS[,CORPUS...]
           [--seed 0] [--epochs 30] [--train-fir] [--aug]

Corpora are data/<corpus>/windows_*.npz (scripts/export_moabb.py): the 41
PROTEUS channels at 120 Hz, 4 s windows, dataset-local labels. ALIGN "ea"
whitens every recording session by its mean covariance (as AlignedNet trains
PROTEUS sessions); "raw" divides each corpus by its global RMS. The net has
one output block per corpus (shared trunk, one head each): the loss is cross
entropy over the window's own block, and batches draw corpora equally often.
Only the trunk transfers; AlignedNet(init=...) starts a fresh 3-class head.
Output: runs/pretrain/<ARCH>_<ALIGN>-<TAG>-s<SEED>.pt (gitignored).
"""

import argparse
import inspect
import json
import time
from pathlib import Path

import braindecode.models as bd
import numpy as np
import torch
from torch import nn

from deep import ARCHS, augment, device
from features import covariances, invsqrtm

ROOT = Path(__file__).resolve().parents[1]


def load_corpus(name, align):
    files = sorted((ROOT / "data" / name).glob("windows_*.npz"))
    assert files, f"no windows for {name}"
    Xs, ys, ss, classes = [], [], [], None
    for f in files:
        with np.load(f) as z:
            assert classes is None or list(z["classes"]) == classes, f
            classes = list(z["classes"])
            Xs.append(z["X"]), ys.append(z["y"])
            ss.append(np.char.add(np.char.add(z["subject"].astype(str), "|"),
                                  z["session"].astype(str)))
    X, y, s = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
    if align == "ea":
        for sess in np.unique(s):
            i = np.flatnonzero(s == sess)
            P = invsqrtm(covariances(X[i].astype(np.float64)).mean(0))
            X[i] = np.einsum("ij,njt->nit", P, X[i]).astype(np.float32)
    else:
        X /= np.float32(np.sqrt((X.astype(np.float64) ** 2).mean()))
    return X, y, classes, len(np.unique(np.char.partition(s, "|")[:, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch"), ap.add_argument("align", choices=("ea", "raw"))
    ap.add_argument("tag"), ap.add_argument("corpora")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-fir", action="store_true")
    ap.add_argument("--aug", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng, gen = np.random.default_rng(a.seed), torch.Generator().manual_seed(a.seed)
    dev, t0 = device(), time.time()
    X, Y, off, meta = [], [], 0, {}
    for c in a.corpora.split(","):
        x, y, classes, n_subj = load_corpus(c, a.align)
        X.append(torch.from_numpy(x)), Y.append(torch.from_numpy(y + off).long())
        meta[c] = dict(windows=len(y), subjects=n_subj, classes=classes, offset=off)
        off += len(classes)
        print(c, meta[c], flush=True)
    sizes = [len(y) for y in Y]
    lo = torch.tensor([meta[c]["offset"] for c in meta])
    hi = torch.tensor([meta[c]["offset"] + len(meta[c]["classes"]) for c in meta])
    cls = getattr(bd, ARCHS[a.arch])  # as deep.build, with one block per corpus
    kw = dict(n_chans=41, n_outputs=off, n_times=480)
    if "sfreq" in inspect.signature(cls).parameters:
        kw["sfreq"] = 120.0
    net = cls(**kw).to(dev)
    if a.train_fir:
        for p in net.spectral_filtering.parameters():
            p.requires_grad_(True)
    steps_per_epoch = int(np.ceil(sum(sizes) / a.batch))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps_per_epoch)
    per = a.batch // len(X)
    net.train()
    for ep in range(a.epochs):
        tot = 0.0
        for _ in range(steps_per_epoch):
            parts = [(c, rng.integers(0, n, per)) for c, n in enumerate(sizes)]
            xb = torch.cat([X[c][j] for c, j in parts]).to(dev)
            yb = torch.cat([Y[c][j] for c, j in parts]).to(dev)
            cb = torch.cat([torch.full((len(j),), c) for c, j in parts]).to(dev)
            if a.aug:
                xb = augment(xb, gen)
            out = net(xb)
            k = torch.arange(off, device=dev)[None]
            own = (k >= lo.to(dev)[cb][:, None]) & (k < hi.to(dev)[cb][:, None])
            loss = nn.functional.cross_entropy(out.masked_fill(~own, -1e4), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep} loss {tot / steps_per_epoch:.4f} {time.time() - t0:.0f}s", flush=True)
    out = ROOT / "runs/pretrain" / f"{a.arch}_{a.align}-{a.tag}-s{a.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()},
                "meta": json.dumps(dict(vars(a), corpora=meta, seconds=round(time.time() - t0)),
                                   default=str)}, out)
    print(json.dumps({"out": str(out), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
