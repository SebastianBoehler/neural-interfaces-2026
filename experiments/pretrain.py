"""Multi-corpus supervised pretraining of the deep families (R10).

Usage: python experiments/pretrain.py ARCH ALIGN TAG CORPUS[,CORPUS...]
           [--seed 0] [--epochs 30] [--train-fir] [--aug] [--rot 0.05]

Corpora are data/<corpus>/windows_*.npz (scripts/export_moabb.py): the 41
PROTEUS channels at 120 Hz, 4 s windows, dataset-local labels. ALIGN "ea"
whitens every recording session by its mean covariance (as AlignedNet trains
PROTEUS sessions); "raw" divides each corpus by its global RMS. The net has
one output block per corpus (shared trunk, one head each): the loss is cross
entropy over the window's own block, and batches draw corpora equally often.
Only the trunk transfers; AlignedNet(init=...) starts a fresh 3-class head.
Output: runs/pretrain/<ARCH>_<ALIGN>-<TAG>-s<SEED>.pt (gitignored).

``--unified C1,C2`` (R17): these corpora share one 4-way block, motor
imagery / subtraction / word generation / rest, each learning its task
against rest inside the same sessions (no block confound). The block's
first three outputs are saved as a 3-class head, which AlignedNet loads.

``--cmix A`` / ``--fsc S`` / ``--mstyle P`` (2 Nov, teacher wave 6): the dg.py
cross-subject levers applied to pretraining (same-class mixing, time scaling,
MixStyle); see dg.py. MixStyle hooks are removed before saving. ``--sr K`` (wave 8):
segment-and-reconstruction augmentation (dg.seg_recombine).
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

import dg
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
    return align_sessions(X, s, align), y, classes, len(np.unique(np.char.partition(s, "|")[:, 0]))


def align_sessions(X, s, align):
    if align == "ea":
        for sess in np.unique(s):
            i = np.flatnonzero(s == sess)
            P = invsqrtm(covariances(X[i].astype(np.float64)).mean(0))
            X[i] = np.einsum("ij,njt->nit", P, X[i]).astype(np.float32)
    else:
        X /= np.float32(np.sqrt((X.astype(np.float64) ** 2).mean()))
    return X


def load_proteus(align):
    """Unlabelled PROTEUS windows for --uda (three window starts): dev participants only, never P04 / P09."""
    Xs = []
    for n in ("proteus_windows", "proteus_windows_start2", "proteus_windows_start3"):
        with np.load(ROOT / "data" / f"{n}.npz") as z:
            subj = z["subject"].astype(str)
            keep = ~np.isin(np.char.rpartition(subj, "/")[:, 2], ["P04", "P09"])
            s = np.char.add(np.char.add(subj[keep], "|"), z["session"].astype(str)[keep])
            Xs.append(align_sessions(z["X"][keep].astype(np.float32), s, align))
    return np.concatenate(Xs)


UNIFIED = {"feet": 0, "hands": 0, "left_hand": 0, "right_hand": 0,  # PROTEUS order + rest
           "subtraction": 1, "S16": 2, "rest": 3, "S32": 3}  # Shin 2018: S16 word gen., S32 baseline


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
    ap.add_argument("--rot", type=float, default=0.0)  # dg.rotate strength (R14)
    ap.add_argument("--unified", default="")  # corpora in the shared 4-way block (R17)
    ap.add_argument("--cmix", type=float, default=0.0)  # dg.class_mix alpha: same-class, cross-subject
    ap.add_argument("--fsc", type=float, default=0.0)  # dg.freq_scale: individual alpha frequency
    ap.add_argument("--mstyle", type=float, default=0.0)  # dg.MixStyle probability per batch
    ap.add_argument("--sr", type=int, default=0)  # dg.seg_recombine segments (wave 8)
    ap.add_argument("--uda", type=float, default=0.0)  # dg.Adversary: corpora vs unlabelled PROTEUS (wave 10)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng, gen = np.random.default_rng(a.seed), torch.Generator().manual_seed(a.seed)
    dev, t0 = device(), time.time()
    uni = [c for c in a.unified.split(",") if c]
    X, Y, off, meta = [], [], 4 if uni else 0, {}
    for c in a.corpora.split(","):
        x, y, classes, n_subj = load_corpus(c, a.align)
        if c in uni:  # shared block at outputs 0-3
            y, o, n = np.array([UNIFIED[classes[i]] for i in y]), 0, 4
        else:
            y, o, n = y + off, off, len(classes)
            off += n
        X.append(torch.from_numpy(x)), Y.append(torch.from_numpy(y).long())
        meta[c] = dict(windows=len(y), subjects=n_subj, classes=classes, offset=o, n=n)
        print(c, meta[c], flush=True)
    sizes = [len(y) for y in Y]
    lo = torch.tensor([meta[c]["offset"] for c in meta])
    hi = torch.tensor([meta[c]["offset"] + meta[c]["n"] for c in meta])
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
    ms = dg.MixStyle(net, gen) if a.mstyle else None
    U = torch.from_numpy(load_proteus(a.align)) if a.uda else None
    adv = dg.Adversary(net, 2, a.uda, a.lr, dev) if a.uda else None
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
            if a.rot:
                xb = dg.rotate(xb, gen, a.rot)
            if a.cmix:  # labels are corpus-offset, so partners share corpus and class
                xb = dg.class_mix(xb, yb, gen, a.cmix)
            if a.fsc:
                xb = dg.freq_scale(xb, gen, a.fsc)
            if a.sr:
                xb = dg.seg_recombine(xb, yb, gen, a.sr)
            if ms:
                ms.on = float(torch.rand(1, generator=gen)) < a.mstyle
            nb = len(xb)
            if adv:  # a quarter batch of unlabelled PROTEUS windows, same augmentation
                ub = U[rng.integers(0, len(U), a.batch // 4)].to(dev)
                xb = torch.cat([xb, dg.rotate(ub, gen, a.rot) if a.rot else ub])
            out = net(xb)
            k = torch.arange(off, device=dev)[None]
            own = (k >= lo.to(dev)[cb][:, None]) & (k < hi.to(dev)[cb][:, None])
            loss = nn.functional.cross_entropy(out[:nb].masked_fill(~own, -1e4), yb)
            if adv:
                loss = loss + adv.loss((torch.arange(len(xb), device=dev) >= nb).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if adv:
                adv.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {ep} loss {tot / steps_per_epoch:.4f} {time.time() - t0:.0f}s", flush=True)
    for h in (ms, adv):
        if h:
            h.remove()
    out = ROOT / "runs/pretrain" / f"{a.arch}_{a.align}-{a.tag}-s{a.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in net.state_dict().items()}
    if uni:  # 3-class head = the shared block's first three outputs
        state = {k: v[:3] if k.startswith("final_layer.") and v.shape[0] == off else v
                 for k, v in state.items()}
    torch.save({"state": state,
                "meta": json.dumps(dict(vars(a), corpora=meta, seconds=round(time.time() - t0)),
                                   default=str)}, out)
    print(json.dumps({"out": str(out), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
