"""Lane R4: one table from the returning summaries (ret/summary_ARM.json: the collector's per-(order, cell, arm) BA table) and the NI-V reports
(out/report_hardcbB_ARM_vs_hardcbB.json). usage: combine.py SUMMARY_DIR NIV_DIR OUT.md ARM...
Returning: gain of ARM over hardcb per (order, cell); P04 / P10 / equal cells / 0.2:0.8, order-only 80 % intervals per person (8 orders), people x orders nested 80 %
for equal cells (two people: descriptive). Scherer B-ret / B-new: the report's paired deltas (people-resampled 80 %). Sealed proxy (UNVERIFIED composition):
0.43 x (0.87 x Scherer B-ret full + 0.13 x mean(P04, P10)); the old weighting 0.2 P04 + 0.8 P10 is shown next to it."""
import json
import sys
from pathlib import Path

import numpy as np

ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
rng = np.random.default_rng(190)


def returning(summary, arm):
    t = json.loads(Path(summary).read_text())["table"]
    g = np.array([[[t[f"{o}|{p}|{v}|{arm}"] - t[f"{o}|{p}|{v}|hardcb"] for v in (2, 3)] for o in ORDERS] for p in ("P04", "P10")])  # (2, 8, 2)
    per = g.mean(2)  # person x order
    pb = [[per[p][rng.integers(0, 8, 8)].mean() for _ in range(4000)] for p in (0, 1)]
    nest = [np.mean([per[p][rng.integers(0, 8, 8)].mean() for p in rng.integers(0, 2, 2)]) for _ in range(4000)]
    ci = lambda x: [float(np.percentile(x, 10)), float(np.percentile(x, 90))]
    return dict(P04=float(per[0].mean()), P10=float(per[1].mean()), eq=float(g.mean()), w2080=float(0.2 * per[0].mean() + 0.8 * per[1].mean()),
                P04_i=ci(pb[0]), P10_i=ci(pb[1]), eq_i=ci(nest), orders_pos=int((per.mean(0) > 0).sum()))


def niv(path):
    d = json.loads(Path(path).read_text())
    f = lambda x: dict(pp=x["pp"], i80=x["i80"], by=x["by_person"])
    return dict(ret_full=f(d["scherer_returning"]["delta"]["primary_full"]), ret_within=f(d["scherer_returning"]["delta"]["secondary_within"]),
                new_full=f(d["scherer_new"]["delta"]["primary_full"]), new_within=f(d["scherer_new"]["delta"]["secondary_within"]),
                abs_ret=d["scherer_returning"]["absolute"], rec=d["scherer_returning"]["recognition"])


S, N, out, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
L = ["| arm | P04 [80 % orders] | P10 [80 % orders] | equal cells [nested 80 %] | 0.2/0.8 | B-ret full [80 %] | B-ret within [80 %] | B-new full / within | sealed proxy (UNVERIFIED) | sealed 0.2/0.8 |", "|" + "---|" * 10]
fmt = lambda v, i: f"{v:+.2f} [{i[0]:+.2f}, {i[1]:+.2f}]"
for a in arms:
    try:
        r = returning(Path(S) / f"summary_{a}.json", a)
    except Exception as e:
        r = None
        print(a, "returning missing:", repr(e)[:100])
    try:
        n = niv(Path(N) / f"report_hardcbB_{a}_vs_hardcbB.json")
    except Exception as e:
        n = None
        print(a, "niv missing:", repr(e)[:100])
    c = [a]
    c += [fmt(r["P04"], r["P04_i"]), fmt(r["P10"], r["P10_i"]), fmt(r["eq"], r["eq_i"]), f"{r['w2080']:+.2f}"] if r else ["-"] * 4
    c += [fmt(n["ret_full"]["pp"], n["ret_full"]["i80"]), fmt(n["ret_within"]["pp"], n["ret_within"]["i80"]),
          f"{n['new_full']['pp']:+.3f} / {n['new_within']['pp']:+.3f}"] if n else ["-"] * 3
    c += [f"{0.43 * (0.87 * n['ret_full']['pp'] + 0.13 * (r['P04'] + r['P10']) / 2):+.3f}" if r and n else "-", f"{0.43 * r['w2080']:+.3f}" if r else "-"]
    L.append("| " + " | ".join(c) + " |")
Path(out).write_text("\n".join(L) + "\n")
print("\n".join(L))
