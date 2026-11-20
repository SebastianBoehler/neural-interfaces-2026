"""Lane W analysis (local, numpy): in-sample arm table and honest nested selection from score_W.json (per-person equal-cell BA, arm x order x person).
usage: python analyze.py score_W.json [out.json]   Rules: evidence/wave_20nov/laneW_corrections.md"""
import json
import sys
from collections import Counter

import numpy as np

ORD = ("full0", "within0")
B = np.random.default_rng(461).integers(0, 10, (10000, 10))
OB = np.random.default_rng(462).integers(0, 2, (10000, 2))


def load(path):
    d = json.load(open(path))["by_person"]
    people = sorted(d["full0"])
    arms = sorted(a for a in d["full0"][people[0]] if a != "cells")
    ba = np.array([[[d[o][p][a] for p in people] for a in arms] for o in ORD])  # order x arm x person
    cells = np.array([d["full0"][p]["cells"] for p in people], float)
    return arms, people, ba, cells


def interval(diff, cells):
    """diff: order x person. Returns point (both orders), people+orders 80 %, people-only 80 % (orders averaged)."""
    w = cells / cells.sum()
    pt = float((diff * w).sum(1).mean())
    pe = np.stack([(diff[:, B] * cells[B]).sum(2) / cells[B].sum(1) for _ in (0,)])[0]  # order x draw
    both = np.take_along_axis(pe.T, OB, 1).mean(1)
    ppl = pe.mean(0)
    f = lambda x: [round(float(v), 3) for v in np.percentile(x, (10, 90))]
    return round(pt, 3), f(both), f(ppl)


def nested(ba, cand, A):
    """cand: arm indices (incl. the no-correction reference). Returns chosen index per (test order, held person) and BA of the choice."""
    idx = {a: i for i, a in enumerate(A)}
    ch = np.zeros((2, ba.shape[2]), int)
    for o in range(2):
        for h in range(ba.shape[2]):
            others = [p for p in range(ba.shape[2]) if p != h]
            score = ba[1 - o][cand][:, others].mean(1)  # selection on the other order, other nine people
            ch[o, h] = cand[int(np.argmax(score))]  # ties go to the first candidate = the reference
    got = np.array([[ba[o, ch[o, h], h] for h in range(ba.shape[2])] for o in range(2)])
    return ch, got


def main(path, out=None):
    A, P, ba, cells = load(path)
    ix = {a: i for i, a in enumerate(A)}
    res = {"in_sample": {}, "nested": {}}
    refs = {"hardcb": ba[:, ix["hardcb"]], "dsall": ba[:, ix["da_s0disp"]], "hcF_l5": ba[:, ix["hcF_l5"]], "stock": ba[:, ix["stock"]]}
    print("arm | vs hardcb | vs ds_all | vs own base | full0/within0 vs ds_all (point)")
    for a in A:
        if "_" not in a or a[:3] not in ("hc_", "da_"):
            continue
        base = "hardcb" if a.startswith("hc_") else "hcF_l5"
        r = {k: interval(ba[:, ix[a]] - v, cells) for k, v in (("hardcb", refs["hardcb"]), ("dsall", refs["dsall"]), ("base", ba[:, ix[base]]))}
        r["orders_vs_dsall"] = [round(float((ba[o, ix[a]] - refs["dsall"][o]) @ (cells / cells.sum())), 3) for o in range(2)]
        res["in_sample"][a] = r
        print(f"{a:16s} {r['hardcb'][0]:+.3f} [{r['hardcb'][1][0]:+.2f},{r['hardcb'][1][1]:+.2f}] | {r['dsall'][0]:+.3f} [{r['dsall'][1][0]:+.2f},{r['dsall'][1][1]:+.2f}] | {r['base'][0]:+.3f} | {r['orders_vs_dsall']}")
    sets = {"hc": ([ix["hardcb"]] + [ix[a] for a in A if a.startswith("hc_")]), "da": ([ix["hcF_l5"]] + [ix[a] for a in A if a.startswith("da_")])}
    sets["pooled"] = [ix["hardcb"]] + [i for k, v in sets.items() if k != "pooled" for i in v if i != ix["hardcb"]]
    sets["pooled"] = [ix["hardcb"], ix["hcF_l5"]] + [i for i in sets["hc"][1:] + sets["da"][1:]]
    print("\nHONEST nested LOPO + cross-order selection (candidates include the uncorrected base)")
    for name, cand in sets.items():
        ch, got = nested(ba, cand, A)
        r = {"vs_hardcb": interval(got - refs["hardcb"], cells), "vs_dsall": interval(got - refs["dsall"], cells), "vs_hcF_l5": interval(got - refs["hcF_l5"], cells),
             "chosen": Counter(A[i] for i in ch.ravel()).most_common(6), "n_cand": len(cand)}
        res["nested"][name] = r
        print(name, "vs hardcb", r["vs_hardcb"], "| vs ds_all", r["vs_dsall"], "| vs hcF_l5", r["vs_hcF_l5"], "\n   chosen", r["chosen"])
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
