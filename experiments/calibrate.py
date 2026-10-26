"""Calibrated fusion: per-family temperature scaling before mixing (R8).

Each family's seed-averaged out-of-fold probabilities on FIT_PROTO get one
temperature T minimising the log-loss of softmax(log p / T) (pooled
windows; no accuracy is optimised). The fitted temperatures are then applied
unchanged on every protocol, families are mixed with their weights and
causally rebalanced as in the packages.
Usage: python experiments/calibrate.py FIT_PROTO PROTOS MEMBER[*W] ...
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
from ensemble import rebalance  # noqa: E402
from mix_probs import member, parse  # noqa: E402

DIRS = "probs,probs_cloud,probs_r5,probs_local,probs_r7,probs_tf,probs_cloud_r8,probs_r8obj".split(",")
d = protocol.load()
dev = ~np.isin(d["subject"], ["Dreyer2026Proteus/P04", "Dreyer2026Proteus/P09"])


def load(proto, spec):
    name, seeds, w = parse(spec)
    src = "all_loso" if proto == "dev_loso_all" else proto
    P, idx, _ = member(name, seeds, src, DIRS)
    if proto == "dev_loso_all":
        keep = dev[idx]
        P, idx = P[keep], idx[keep]
    return P, idx, w


def temper(P, T):
    z = np.log(np.clip(P, 1e-9, 1)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_T(P, y):
    nll = lambda logT: -np.mean(np.log(np.clip(temper(P, np.exp(logT))[np.arange(len(y)), y], 1e-12, 1)))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


fit_proto, protos, specs = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
Ts = []
for spec in specs:
    P, idx, _ = load(fit_proto, spec)
    Ts.append(fit_T(P, d["y"][idx]))
print("temperatures", {s.split("*")[0]: round(t, 2) for s, t in zip(specs, Ts)})
for proto in protos:
    row = []
    for cal in (False, True):
        ms = [load(proto, s) for s in specs]
        idx = ms[0][1]
        tot = sum(w for _, _, w in ms)
        P = sum(w / tot * (temper(p, T) if cal else p) for (p, _, w), T in zip(ms, Ts))
        st = d["stream"][idx]
        for s in dict.fromkeys(st):
            P[st == s] = rebalance(P[st == s], 1, 5, 30)
        mask = np.zeros(len(d["y"]), bool)
        mask[idx] = True
        row.append(100 * protocol.score(d["y"][mask], P.argmax(1)[np.argsort(idx)], d, mask)
                   ["balanced_accuracy_cell_mean"])
    print(f"{proto:14s} raw {row[0]:.2f}  calibrated {row[1]:.2f}  ({row[1] - row[0]:+.2f})")
