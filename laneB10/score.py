"""B10 NETBCI stress scoring (local, from the small ret.npz files): MI recall, non-MI specificity, BA2, recognition and routed share per (variant, person, day, order).
usage: score.py OUT_DIR DEST.json. Deltas vs hardcbB paired per person; 80 % intervals resample the 19 people (10 000 draws)."""
import json
import sys
from pathlib import Path

import numpy as np

out, dest = Path(sys.argv[1]), sys.argv[2]
V = ["v66_cp1_cap250", "hardcbB", "hardcbB_R1", "hardcbB_R2", "hardcbB_R5", "hardcbB_R1a", "hardcbB_R3", "hardcbB_R4", "v66_cp1_cap250_new", "hardcbB_new"]
rng = np.random.default_rng(1010)


def load(v):
    """arrays (person, day 2-4, order) of mi recall, specificity, ba2, rec, routed share; NaN where a task is missing"""
    A = np.full((5, 19, 3, 8), np.nan)
    for k in range(19):
        f = out / f"{v}_k{k}" / "ret.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            tags = sorted({t.rsplit("|", 1)[0] for t in z.files}, key=lambda t: (t.split("|")[0], int(t.split("|")[1]), int(t.split("|")[2])))
            for t in tags:
                sc, sd, d = t.split("|")
                o = (0 if sc == "full" else 4) + int(sd) - 72000
                y, lab = z[t + "|y"], z[t + "|labels"]
                mi, sp = (lab[y == 1] == 0).mean(), (lab[y == 0] != 0).mean()
                rec, routed = z[t + "|meta"][:2]
                A[:, k, int(d) - 2, o] = [100 * mi, 100 * sp, 50 * (mi + sp), rec, routed / len(y)]
    return A


def boot(x):
    x = x[~np.isnan(x)]
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]
    return float(x.mean()), [float(np.percentile(b, 10)), float(np.percentile(b, 90))], len(x)


data = {v: load(v) for v in V if any((out / f"{v}_k{k}" / "ret.npz").exists() for k in range(19))}
res = {"abs": {}, "delta_vs_hardcbB": {}, "enrol_vs_new": {}, "recognition": {}, "per_person_day": {}}
for v, A in data.items():
    ba = A[2]
    per = np.nanmean(ba, axis=2)  # person x day
    res["abs"][v] = dict(pooled=boot(np.nanmean(per, 1))[0], by_day=[float(np.nanmean(per[:, d])) for d in range(3)],
                         mi=float(np.nanmean(A[0])), spec=float(np.nanmean(A[1])), n_people=int((~np.isnan(per[:, 0])).sum()))
    rec, routed = A[3], A[4]
    res["recognition"][v] = dict(share_recognised=float(np.nanmean(rec >= 0)), median_latency=float(np.nanmedian(rec[rec >= 0])) if (rec >= 0).any() else None,
                                 routed_share_all=float(np.nanmean(routed)), by_day_recognised=[float(np.nanmean(rec[:, d] >= 0)) for d in range(3)],
                                 by_day_routed=[float(np.nanmean(routed[:, d])) for d in range(3)])
ref = "hardcbB"
for v, A in data.items():
    if v == ref or ref not in data:
        continue
    d = np.nanmean(A[2] - data[ref][2], axis=2)  # person x day
    res["delta_vs_hardcbB"][v] = dict(pooled=boot(np.nanmean(d, 1)), by_day=[boot(d[:, i]) for i in range(3)])
    res["per_person_day"][v] = np.round(d, 2).tolist()
for v in ("v66_cp1_cap250", "hardcbB"):
    if v in data and v + "_new" in data:
        d = np.nanmean(data[v][2] - data[v + "_new"][2], axis=2)
        res["enrol_vs_new"][v] = dict(pooled=boot(np.nanmean(d, 1)), by_day=[boot(d[:, i]) for i in range(3)])
Path(dest).write_text(json.dumps(res, indent=1))
print("abs BA2 (pooled / by day / MI recall / spec / people):")
for v, e in res["abs"].items():
    print(f"  {v:22s} {e['pooled']:.2f} {np.round(e['by_day'], 2).tolist()} MI {e['mi']:.1f} spec {e['spec']:.1f} n {e['n_people']}")
print("recognition:")
for v, e in res["recognition"].items():
    print(f"  {v:22s} recognised {e['share_recognised']:.2f} median latency {e['median_latency']} routed {e['routed_share_all']:.2f} by day rec {np.round(e['by_day_recognised'], 2).tolist()} routed {np.round(e['by_day_routed'], 2).tolist()}")
print("delta vs hardcbB (pp, 80 % people):")
for v, e in res["delta_vs_hardcbB"].items():
    print(f"  {v:22s} pooled {e['pooled'][0]:+.2f} [{e['pooled'][1][0]:+.2f}, {e['pooled'][1][1]:+.2f}] days " + " ".join(f"{x[0]:+.2f}" for x in e["by_day"]))
print("enrolled minus new (pp):")
for v, e in res["enrol_vs_new"].items():
    print(f"  {v:22s} pooled {e['pooled'][0]:+.2f} [{e['pooled'][1][0]:+.2f}, {e['pooled'][1][1]:+.2f}] days " + " ".join(f"{x[0]:+.2f}" for x in e["by_day"]))
