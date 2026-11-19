"""Paired people+orders bootstrap between two scored runs.

python compare.py RUN_A RUN_B [--draws 10000] [--json out.json]     reports A minus B (percentage points).
A person's value = mean of that person's official cells; a draw resamples people with replacement and, independently,
orders with replacement (orders both runs share; if the runs share none, each side is averaged over its own orders first
and only people are resampled - flagged). Held-new = all 10 people. Returning = P04 and P10 ONLY (two people, later
sessions): its interval is wide and the sealed 0.43 weight rests on two people. Sealed-weighted = 0.57 new + 0.43 returning
(Sealed mix is ~57 % new people; returning people ~0.2 P04 + 0.8 P10 is NOT applied here: equal people).
"""
import argparse
import json
from pathlib import Path

import numpy as np

W_NEW = 0.57


def person_table(run, mode):
    """{order: {person: value}}"""
    with np.load(Path(run) / "cells.npz", allow_pickle=False) as z:
        m = z["mode"] == mode
        out = {}
        for o in sorted(set(z["order"][m])):
            mo = m & (z["order"] == o)
            out[int(o)] = {p: float(z["ba"][mo & (z["person"] == p)].mean()) for p in sorted(set(z["person"][mo]))}
        head = {int(o): float(z["ba"][m & (z["order"] == o)].mean()) for o in out}
    return out, head


def delta_matrix(ta, tb):
    """(orders, people) paired differences; shared orders or, failing that, per-side means over orders."""
    people = sorted(set.intersection(*(set(v) for v in ta.values()), *(set(v) for v in tb.values())))
    shared = sorted(set(ta) & set(tb))
    if shared:
        return np.array([[ta[o][p] - tb[o][p] for p in people] for o in shared]), people, shared, False
    ma = np.array([[ta[o][p] for p in people] for o in ta]).mean(0)
    mb = np.array([[tb[o][p] for p in people] for o in tb]).mean(0)
    return (ma - mb)[None], people, [], True


def draws(delta, rng, n):
    no, npers = delta.shape
    oi, pi = rng.integers(no, size=(n, no)), rng.integers(npers, size=(n, npers))
    return delta[oi[:, :, None], pi[:, None, :]].mean((1, 2))


def iv(x, level):
    lo = (1 - level) / 2 * 100
    return [round(float(np.percentile(x, lo)), 3), round(float(np.percentile(x, 100 - lo)), 3)]


def summarize(x):
    return dict(pp=round(float(x.mean()), 3), i80=iv(x, .8), i95=iv(x, .95), p_pos=round(float((x > 0).mean()), 3))


def compare(run_a, run_b, n=10000, seed=19):
    rng = np.random.default_rng(seed)
    res, boots = {}, {}
    for mode in ("new", "ret"):
        try:
            ta, ha = person_table(run_a, mode)
            tb, hb = person_table(run_b, mode)
        except Exception:
            continue
        if not ta or not tb:
            continue
        d, people, shared, flat = delta_matrix(ta, tb)
        boots[mode] = draws(d, rng, n)
        res[mode] = dict(point_pp=round(float(d.mean()), 3), n_people=len(people), shared_orders=shared,
                         orders_not_shared=flat, cellmean_A=ha, cellmean_B=hb, by_person=dict(zip(people, d.mean(0).round(3).tolist())),
                         **{k: v for k, v in summarize(boots[mode]).items()})
    if len(boots) == 2:
        res["sealed"] = dict(w_new=W_NEW, note="returning covers only P04/P10", **summarize(W_NEW * boots["new"] + (1 - W_NEW) * boots["ret"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = compare(a.a, a.b, a.draws)
    for k, v in r.items():
        print(k, json.dumps({x: v[x] for x in v if x not in ("by_person", "cellmean_A", "cellmean_B")}))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
