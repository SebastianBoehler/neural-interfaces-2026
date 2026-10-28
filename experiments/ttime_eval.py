"""Label-free test-time training by information maximisation (T-TIME-style, R12).

T-TIME (Li et al., arXiv 2412.07228): after predicting window t of a
session, take one gradient step on the last ``B`` windows of the session
(t included) minimising the mean prediction entropy minus the entropy of
the mean prediction (confident yet class-diverse outputs). No labels, a
fresh copy of the trained net and optimiser per session, causal: window t
is predicted before any update that uses it.

For every fold a net is trained once; every variant (and the frozen
control) predicts the same test sessions with it, saved as
CONFIG_ttVARIANT-PROTOCOL-sSEED.npz (control: CONFIG_tt0) for paired
comparison with mix_probs.py.
Usage: python experiments/ttime_eval.py CONFIG PROTOCOLS SEED [VARIANT ...]
"""
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import protocol  # noqa: E402
import run_cv  # noqa: E402

# name -> (parameters adapted, Adam learning rate, buffer B, steps per window)
VARIANTS = {"0": None,
            "a1": ("all", 1e-4, 8, 1), "a3": ("all", 3e-4, 8, 1), "a10": ("all", 1e-3, 8, 1),
            "n10": ("norm", 1e-3, 8, 1), "a3b16": ("all", 3e-4, 16, 1)}


def im_loss(logits):
    p = torch.softmax(logits, 1)
    cond = -(p * torch.log(p + 1e-8)).sum(1).mean()
    m = p.mean(0)
    return cond + (m * torch.log(m + 1e-8)).sum()  # cond - H(mean)


def adapt_stream(net, X, spec):
    """Probabilities of one session's windows X (aligned, in order)."""
    if spec is None:
        with torch.no_grad():
            return torch.softmax(net(X), 1).cpu().double().numpy()
    which, lr, B, steps = spec
    net = copy.deepcopy(net).eval()  # BN keeps training statistics; no dropout
    if which == "norm":
        params = [p for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
                  for p in m.parameters()]
    else:  # parameters trained with the net (a frozen FIR bank stays frozen)
        params = [p for p in net.parameters() if p.requires_grad]
    for p in net.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    out = []
    for t in range(len(X)):
        with torch.no_grad():
            out.append(torch.softmax(net(X[t:t + 1]), 1).cpu().double())
        if t >= 1:  # need two windows for a marginal
            for _ in range(steps):
                opt.zero_grad()
                im_loss(net(X[max(0, t + 1 - B):t + 1])).backward()
                opt.step()
    return torch.cat(out).numpy()


def main():
    cfg, protos, seed = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
    names = sys.argv[4:] or list(VARIANTS)
    run_cv.SEED[0] = seed
    out = ROOT / os.environ.get("NI26_PROBS", "runs/experiments/probs_ttime")
    out.mkdir(parents=True, exist_ok=True)
    d = protocol.load()
    make = run_cv.resolve(cfg)
    for proto in protos:
        res, idx_all, secs = {v: [] for v in names}, [], {v: 0.0 for v in names}
        for name, tr, te in protocol.folds(d, proto):
            m = make(d)
            m.fit(np.flatnonzero(tr))
            for s in dict.fromkeys(d["stream"][te]):
                rows = np.flatnonzero(te & (d["stream"] == s))
                X = torch.from_numpy(m._aligned(rows, causal=True)).to(m.dev)
                for v in names:
                    t0 = time.time()
                    res[v].append(adapt_stream(m.net, X, VARIANTS[v]))
                    secs[v] += time.time() - t0
                idx_all.append(rows)
            print(cfg, proto, name, flush=True)
        idx = np.concatenate(idx_all)
        order = np.argsort(idx)  # global row order, as protocol.evaluate saves
        mask = np.zeros(len(d["y"]), bool)
        mask[idx] = True
        for v in names:
            P = np.concatenate(res[v])
            np.savez(out / f"{cfg}_tt{v}-{proto}-s{seed}.npz", proba=P[order], index=idx[order])
            sc = protocol.score(d["y"][mask], P.argmax(1)[order], d, mask)
            print(f"{cfg} {proto} s{seed} tt{v} cell={sc['balanced_accuracy_cell_mean']:.4f} "
                  f"{1000 * secs[v] / len(idx):.1f} ms/window", flush=True)


if __name__ == "__main__":
    main()
