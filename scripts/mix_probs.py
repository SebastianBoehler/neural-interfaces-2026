"""Score offline mixes of saved per-window probabilities (rounds 4+ rule).

Usage: python scripts/mix_probs.py PROTOCOLS MEMBER [MEMBER ...]
       [--alpha 1] [--k 5] [--w 30] [--dirs probs,probs_cloud,...]
MEMBER is ``NAME[:SEEDS][*WEIGHT]``, e.g. ``v2``, ``fbcnet_raw:0-4*2`` or
``shallow_raw@scherer2015~n~w0.1``. A member's saved seeds (first directory
wins per seed, so no seed is counted twice) are averaged; members are mixed
with their weights (default 1, normalised) and causally rebalanced per
session with ``experiments.ensemble.rebalance(P, alpha, k, W)``, as in the
packages. Prints the cell mean per protocol; ``all_loso`` also lists P04 and
P09, the only participants no selection round has scored.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import protocol  # noqa: E402
from ensemble import rebalance  # noqa: E402

DIRS = "probs,probs_cloud,probs_r5,probs_local,probs_r7"


def parse(spec):
    spec, _, w = spec.partition("*")
    name, _, seeds = spec.partition(":")
    if seeds:
        lo, _, hi = seeds.partition("-")
        seeds = {str(s) for s in range(int(lo), int(hi or lo) + 1)}
    return name, seeds or None, float(w or 1)


def member(name, seeds, proto, dirs):
    runs = {}
    for d in dirs:
        for f in sorted((ROOT / "runs/experiments" / d).glob(f"{name}-{proto}-s*.npz")):
            s = f.stem.rsplit("-s", 1)[1]
            if seeds is None or s in seeds:
                runs.setdefault(s, f)
    assert runs, f"no probabilities for {name} / {proto}"
    loaded = [np.load(f) for f in runs.values()]
    idx = loaded[0]["index"]
    assert all((z["index"] == idx).all() for z in loaded)
    return np.mean([z["proba"] for z in loaded], 0), idx, sorted(runs, key=int)


def mix(d, proto, specs, dirs, alpha, k, w):
    ms = [(member(n, s, proto, dirs), wt) for n, s, wt in map(parse, specs)]
    idx = ms[0][0][1]
    assert all((m[1] == idx).all() for m, _ in ms)
    tot = sum(wt for _, wt in ms)
    P = sum(wt / tot * m[0] for m, wt in ms)
    streams = d["stream"][idx]
    if alpha:
        P = np.concatenate([rebalance(P[streams == s], alpha, k, w)
                            for s in dict.fromkeys(streams)])
        idx = np.concatenate([idx[streams == s] for s in dict.fromkeys(streams)])
    mask = np.zeros(len(d["y"]), bool)
    mask[idx] = True
    order = np.argsort(idx)  # protocol.score groups by mask order
    pred = P.argmax(1)[order]
    res = {"cell_mean": round(100 * protocol.score(d["y"][mask], pred, d, mask)
                              ["balanced_accuracy_cell_mean"], 2),
           "seeds": {n: m[2] for (m, _), (n, _, _) in zip(ms, map(parse, specs))}}
    if proto == "all_loso":
        for p in ("P04", "P09"):
            sub = mask & (d["subject"] == f"Dreyer2026Proteus/{p}")
            res[p] = round(100 * protocol.score(d["y"][sub], pred[sub[mask]], d, sub)
                           ["balanced_accuracy_cell_mean"], 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("protocols")
    ap.add_argument("members", nargs="+")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--w", type=int, default=30)
    ap.add_argument("--dirs", default=DIRS)
    args = ap.parse_args()
    d = protocol.load()
    out = {p: mix(d, p, args.members, args.dirs.split(","), args.alpha, args.k, args.w)
           for p in args.protocols.split(",")}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
