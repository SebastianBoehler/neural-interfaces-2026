"""Scherer B-ret on the Leonardo orders (explicit seeds 71008-71023, 16 full + 16 within): BA per (person, scheme, seed) of each variant, paired deltas vs hardcbB,
people-resampled 80 % intervals (people x orders). usage: score.py OUT_DIR WINDOWS_NPZ OUT_JSON VARIANT..."""
import json
import sys
from pathlib import Path

import numpy as np

out, win, dest, variants = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
with np.load(win, allow_pickle=False) as z:
    y = z["y"]
names = sorted(set(np.load(win, allow_pickle=False)["subject"]))
rng = np.random.default_rng(190)


def ba(v):
    """(scheme: 2, seed: 16, person: 9) balanced accuracy, plus recognition age and routed windows"""
    B = np.full((2, 16, 9), np.nan)
    R = np.full((2, 16, 9, 2), np.nan)
    for k in range(9):
        f = out / f"{v}_k{k}" / "ret.npz"
        if not f.exists():
            continue
        with np.load(f) as d:
            for si, s in enumerate(("full", "within")):
                for i in range(16):
                    tag = f"{s}|s{71008 + i}|{names[k]}|2"
                    rows, lab = d[tag + "|rows"], d[tag + "|labels"]
                    t = y[rows]
                    B[si, i, k] = 100 * np.mean([(lab[t == c] == c).mean() for c in range(3)])
                    R[si, i, k] = d[tag + "|meta"][:2]
    return B, R


base, Rb = ba("hardcbB")
res = dict(base_abs=[float(np.nanmean(base[s])) for s in range(2)], variants={})
for v in variants:
    B, R = ba(v)
    ok = ~np.isnan(B).any(axis=(0, 1)) & ~np.isnan(base).any(axis=(0, 1))
    d = (B - base)[:, :, ok]  # scheme x seed x person
    per_person = d.mean(1)  # scheme x person
    e = {}
    for si, s in enumerate(("full", "within")):
        boots = [per_person[si][rng.integers(0, ok.sum(), ok.sum())].mean() for _ in range(10000)]
        e[s] = dict(pp=float(per_person[si].mean()), i80=[float(np.percentile(boots, 10)), float(np.percentile(boots, 90))],
                    by_person=per_person[si].round(2).tolist(), n_people=int(ok.sum()))
    allp = d.mean((0, 1))
    boots = [allp[rng.integers(0, ok.sum(), ok.sum())].mean() for _ in range(10000)]
    e["pooled"] = dict(pp=float(allp.mean()), i80=[float(np.percentile(boots, 10)), float(np.percentile(boots, 90))])
    e["abs"] = [float(np.nanmean(B[s][:, ok])) for s in range(2)]
    e["rec_age_mean"] = float(np.nanmean(R[..., 0])); e["routed_mean"] = float(np.nanmean(R[..., 1]))
    res["variants"][v] = e
Path(dest).write_text(json.dumps(res, indent=1))
for v, e in res["variants"].items():
    print(f"{v:14s} full {e['full']['pp']:+.3f} {e['full']['i80']}  within {e['within']['pp']:+.3f} {e['within']['i80']}  pooled {e['pooled']['pp']:+.3f} {e['pooled']['i80']}  abs {e['abs']}")
