"""Paired variant contrasts on team C eval files: A minus B per order and pooled over orders, people-resampled 80 %.

python cleanslate/teamC/paired.py DIR_A:VAR_A DIR_B:VAR_B [--orders 0,1] [--people P01,...]
Person value = mean over the person's cells (same rows on both sides, asserted); pooled = mean over orders per person.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import ROOT, cells  # noqa: E402


def person_cells(z, var, k, d):
    return cells(d, z[f"rows|{k}"], z[f"{var}|{k}|label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--orders", default="0,1")
    ap.add_argument("--people", default=",".join(f"P{i:02}" for i in range(1, 11)))
    x = ap.parse_args()
    d = np.load(ROOT / "data/proteus_windows.npz")
    (da, va), (db, vb) = x.a.split(":"), x.b.split(":")
    people, orders = x.people.split(","), x.orders.split(",")
    D = np.zeros((len(people), len(orders)))
    for i, p in enumerate(people):
        za, zb = np.load(Path(da) / f"eval_{p}.npz"), np.load(Path(db) / f"eval_{p}.npz")
        for j, k in enumerate(orders):
            assert np.array_equal(za[f"rows|{k}"], zb[f"rows|{k}"])
            ca, cb = person_cells(za, va, k, d), person_cells(zb, vb, k, d)
            D[i, j] = 100 * np.mean([ca[q] - cb[q] for q in ca])
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(people), (10000, len(people)))
    for j, k in enumerate(orders):
        print(f"order {k}: {D[:, j].mean():+.2f} {np.percentile(D[pick, j].mean(1), [10, 90]).round(2)}")
    pool = D.mean(1)
    print(f"pooled: {pool.mean():+.2f} {np.percentile(pool[pick].mean(1), [10, 90]).round(2)}  per person "
          + " ".join(f"{p}:{v:+.1f}" for p, v in zip(people, pool)))


if __name__ == "__main__":
    main()
