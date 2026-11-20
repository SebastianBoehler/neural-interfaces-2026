"""Collect task npz files into official cell scores per config (harness cell_table) and a leaderboard.

python collect.py RUN_DIR --stage s1 [--out lb.json]    RUN_DIR holds task_<held>_<chunk>.npz (+ .json) files.
Writes RUN_DIR/<stage>_cells.npz (cid -> arrays), prints leaderboard sorted by held-new cell mean over the people present.
Intervals: bootstrap over people and orders (10k draws, person value = mean of its cells), 80 % and 95 %.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
import data as D  # noqa: E402


def load_tasks(run, d):
    """cid -> {(order, person): pred-by-row, mask} and secs/windows per cid."""
    store, tim, names = {}, {}, {}
    for f in sorted(glob.glob(f"{run}/task_*.npz")):
        meta = json.load(open(f + ".json"))
        for c in meta["cfgs"]:
            names[c["cid"]] = (c["name"], c["cfg"])
        with np.load(f) as z:
            for k in z.files:
                m = re.fullmatch(r"(\w+)\|rows_(\d+)_(\w+)", k)
                if not m:
                    continue
                h, order, ses = m.group(1), int(m.group(2)), m.group(3)
                st = store.setdefault(h, {}).setdefault((order, meta["held"]), (np.full(len(d["y"]), -1, np.int8), np.zeros(len(d["y"]), bool)))
                rows = z[k]
                st[0][rows] = z[f"{h}|labels_{order}_{ses}"]
                st[1][rows] = True
                t = tim.setdefault(h, [0.0, 0])
                t[0] += float(z[f"{h}|secs_{order}_{ses}"])
                t[1] += len(rows)
    return store, tim, names


def cells_for(store, d):
    out = {}
    for h, per in store.items():
        recs = []
        for (order, person), (pred, mask) in sorted(per.items()):
            recs += D.cell_table(d, pred, mask, order, "new")
        out[h] = recs
    return out


def person_matrix(recs, people, orders):
    M = np.full((len(people), len(orders)), np.nan)
    for i, p in enumerate(people):
        for j, o in enumerate(orders):
            v = [r["ba"] for r in recs if r["person"] == p and r["order"] == o]
            if v:
                M[i, j] = np.mean(v)
    return M


def boot(M, ref=None, n=10000, seed=0):
    """Mean of (M - ref) over people x orders, resampling people and orders; returns mean, (lo80, hi80), (lo95, hi95)."""
    D_ = M if ref is None else M - ref
    rng = np.random.default_rng(seed)
    P, O = D_.shape
    vals = np.empty(n)
    for b in range(n):
        vals[b] = D_[np.ix_(rng.integers(0, P, P), rng.integers(0, O, O))].mean()
    q = lambda a, b: (float(np.percentile(vals, a)), float(np.percentile(vals, b)))
    return float(D_.mean()), q(10, 90), q(2.5, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stage", default="s")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ref", default="A1", help="config name whose cid is the no-TTT reference (A1 = prefix BN, no gradient TTT)")
    a = ap.parse_args()
    d = D.load(a.data)
    store, tim, names = load_tasks(a.run, d)
    cells = cells_for(store, d)
    people = sorted({p for per in store.values() for (_, p) in per})
    orders = sorted({o for per in store.values() for (o, _) in per})
    mats = {h: person_matrix(r, people, orders) for h, r in cells.items()}
    full = {h: m for h, m in mats.items() if not np.isnan(m).any()}
    ref = next((h for h, (n, _) in names.items() if n == a.ref and h in full), None)
    rows = []
    for h, m in full.items():
        mean, i80, i95 = boot(m)
        g = boot(m, full[ref])[:3] if ref else (None, None, None)
        rows.append(dict(cid=h, name=names[h][0], ba=mean, i80=i80, i95=i95, gain_vs_ref=g[0], gain80=g[1], gain95=g[2],
                         ms_per_window=1e3 * tim[h][0] / tim[h][1], people=len(people), orders=len(orders), cfg=names[h][1],
                         by_person={p: float(np.nanmean(m[i])) for i, p in enumerate(people)}))
    rows.sort(key=lambda r: -r["ba"])
    json.dump(dict(stage=a.stage, people=people, orders=orders, ref=a.ref, rows=rows), open(f"{a.run}/{a.stage}_leaderboard.json", "w"), indent=1)
    np.savez(f"{a.run}/{a.stage}_pm.npz", **{h: m for h, m in full.items()})
    print(f"{len(rows)} configs, people {people}, orders {orders}, ref {a.ref}")
    for r in rows[:15]:
        print(f"{r['name']:<22}{r['cid']} BA {r['ba']:.2f} [{r['i80'][0]:.1f},{r['i80'][1]:.1f}] gain {r['gain_vs_ref']:+.2f} {r['ms_per_window']:.0f} ms/win")


if __name__ == "__main__":
    main()
