"""Score causal class rebalancing on saved per-window probabilities.

Usage: python scripts/rebalance_probs.py OUT.json PROTOCOL [PROTOCOL ...]
Every scoring cell has balanced classes, so a model that drifts towards one
class on a new person loses balanced accuracy. Rebalancing divides window
t's probabilities by m_t ** alpha, where m_t is the mean predicted
probability of the session's windows 0..t-1 blended with a uniform prior
weighted as ``k`` windows. It uses no labels and only earlier windows.
Base model: the round-2 ensemble (v2 + aligned x5 + raw x5, equal thirds).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import protocol  # noqa: E402

DIRS = (ROOT / "runs/experiments/probs", ROOT / "runs/experiments/probs_cloud")


def member(name, proto):
    runs = [np.load(f) for d in DIRS for f in sorted(d.glob(f"{name}-{proto}-s*.npz"))]
    if not runs:
        return None, None
    idx = runs[0]["index"]
    assert all((r["index"] == idx).all() for r in runs)
    return np.mean([r["proba"] for r in runs], 0), idx


def base(proto):
    parts = [member(n, proto) for n in ("v2", "shallow_ea", "shallow_raw")]
    if any(p is None for p, _ in parts):
        return None, None
    assert all((i == parts[0][1]).all() for _, i in parts)
    return np.mean([p for p, _ in parts], 0), parts[0][1]


def rebalance(P, streams, alpha, k):
    out = np.empty_like(P)
    for s in dict.fromkeys(streams):
        rows = np.flatnonzero(streams == s)
        seen = np.zeros(P.shape[1])
        for n, r in enumerate(rows):
            m = (k / P.shape[1] + seen) / (k + n)
            q = P[r] / m ** alpha
            out[r] = q / q.sum()
            seen += P[r]
    return out


def main():
    out_path, protos = sys.argv[1], sys.argv[2:]
    d = protocol.load()
    res = {}
    for proto in protos:
        P, idx = base(proto)
        if P is None:
            print(proto, "missing members")
            continue
        mask = np.zeros(len(d["y"]), bool)
        mask[idx] = True
        y, streams = d["y"][idx], d["stream"][idx]
        for alpha in (0.0, 0.5, 1.0):
            for k in ((5.0,) if alpha == 0 else (2.0, 5.0, 20.0)):
                Q = rebalance(P, streams, alpha, k)
                sc = protocol.score(y, Q.argmax(1), d, mask)
                res[f"{proto}|a{alpha}|k{k:g}"] = round(sc["balanced_accuracy_cell_mean"], 4)
                print(f"{proto:12s} alpha={alpha} k={k:4g} "
                      f"cell={sc['balanced_accuracy_cell_mean']:.4f}")
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
