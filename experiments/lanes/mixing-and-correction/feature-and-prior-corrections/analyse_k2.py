"""Lane K2 analysis (numpy only; cells.json from chain/score_k2.py). Rule registered in evidence/wave_20nov/laneK2_ideas456.md.

  python analyse_k2.py cells.json     spec-weight arms vs base and vs matched controls, nested weight-scale selection, P04/P10, people and people+orders 80 % intervals
Arms in cells.json: `sys|name`. Nested selection: held-out person j in order o takes the scale (0.5, 1, 2, else 0) with the best mean train delta over the OTHER order's other nine people."""
import json
import sys
from pathlib import Path

import numpy as np

ORDERS = ("full0", "within0")
SYSTEMS = ("hardcb", "ds_l5")
SCALES = ("0.5", "1.0", "2.0")
FORMS = {"x4c": "idea 4 cond", "x4m": "idea 4 marg", "x4p": "idea 4 perm (mean of 3 seeds)", "x6g": "idea 6 gen", "x6a": "idea 6 abs"}


def parse(path):
    c = {}
    for o, arms in json.loads(Path(path).read_text())["cells"].items():
        c[o] = {k: (np.array([v[p][0] for p in sorted(v)]), np.array([v[p][1] for p in sorted(v)]), sorted(v)) for k, v in arms.items()}
    return c


def cells(c, o, sysm, form, scale):
    """per-person (sum BA, cells) of a form at a scale; x4p averages the three permutation seeds; 'base' and scale 0 give the base."""
    if scale == "0" or form == "base":
        return c[o][f"{sysm}|base"][0]
    names = [f"{sysm}|{form}{s}_s{scale}" for s in ("0", "1", "2")] if form == "x4p" else [f"{sysm}|{form}_s{scale}"]
    return np.mean([c[o][n][0] for n in names], axis=0)


def arm_cells(c, sysm, spec):
    """spec = (form, scale) or ('nest', form[, control]) -> {order: per-person sums}; nested picks per person and order."""
    if spec[0] != "nest":
        return {o: cells(c, o, sysm, *spec) for o in ORDERS}
    form = spec[1]; out = {}
    for o in ORDERS:
        other = [x for x in ORDERS if x != o][0]; n = c[o][f"{sysm}|base"][1]; s = np.empty(10)
        for j in range(10):
            tr = np.arange(10) != j; best, pick = 0.0, "0"
            for sc in SCALES:
                d = 100 * ((cells(c, other, sysm, form, sc) - cells(c, other, sysm, "base", "0"))[tr].sum() / c[other][f"{sysm}|base"][1][tr].sum())
                if d > best:
                    best, pick = d, sc
            s[j] = cells(c, o, sysm, form, pick)[j]
        out[o] = s
    return out


def delta(c, sysm, a, b):
    """per order, per person (cell-sum difference), plus cell counts."""
    A, B = arm_cells(c, sysm, a), arm_cells(c, sysm, b)
    return {o: A[o] - B[o] for o in ORDERS}, c["full0"][f"{sysm}|base"][1]


def summarize(d, n, who):
    pp = np.random.default_rng(461).integers(0, 10, (10000, 10)); oo = np.random.default_rng(462).integers(0, 2, (10000, 2))
    mean = np.mean([d[o] for o in ORDERS], axis=0)
    ppl = np.percentile(100 * mean[pp].sum(1) / n[pp].sum(1), (10, 90))
    arr = np.stack([d[o] for o in ORDERS])
    po = np.percentile([100 * arr[oo[i]][:, pp[i]].mean(0).sum() / n[pp[i]].sum() for i in range(10000)], (10, 90))
    i4, i10 = who.index("P04"), who.index("P10")
    p4, p10 = 100 * mean[i4] / n[i4], 100 * mean[i10] / n[i10]
    f = lambda o: 100 * d[o].sum() / n.sum()
    return (f"{100 * mean.sum() / n.sum():+.2f} [{ppl[0]:+.2f},{ppl[1]:+.2f}] po [{po[0]:+.2f},{po[1]:+.2f}] (full0 {f('full0'):+.2f} / within0 {f('within0'):+.2f}) "
            f"P04 {p4:+.2f} P10 {p10:+.2f} 0.2/0.8(UNVERIFIED) {0.2 * p4 + 0.8 * p10:+.2f}")


def report(path):
    c = parse(path); who = [p.rsplit("/", 1)[-1] for p in c["full0"]["hardcb|base"][2]]
    for sysm in SYSTEMS:
        ones = np.ones(10, bool)
        print(f"\n## {sysm}: base {np.mean([100 * c[o][f'{sysm}|base'][0].sum() / c[o][f'{sysm}|base'][1].sum() for o in ORDERS]):.2f}")
        for lab, a, b in (
                ("idea 4 cond vs base (s1)", ("x4c", "1.0"), ("base", "0")), ("idea 4 cond vs perm (s1)", ("x4c", "1.0"), ("x4p", "1.0")), ("idea 4 cond vs marg (s1)", ("x4c", "1.0"), ("x4m", "1.0")),
                ("idea 4 marg vs base (s1)", ("x4m", "1.0"), ("base", "0")), ("idea 4 perm vs base (s1)", ("x4p", "1.0"), ("base", "0")),
                ("idea 4 cond vs base nested-scale", ("nest", "x4c"), ("base", "0")), ("idea 4 cond nested vs perm nested", ("nest", "x4c"), ("nest", "x4p")),
                ("idea 4 oracle ceiling vs base", ("x4o", "1.0"), ("base", "0")),
                ("idea 6 gen vs base (s1)", ("x6g", "1.0"), ("base", "0")), ("idea 6 gen vs abs (s1)", ("x6g", "1.0"), ("x6a", "1.0")), ("idea 6 abs vs base (s1)", ("x6a", "1.0"), ("base", "0")),
                ("idea 6 gen vs base nested-scale", ("nest", "x6g"), ("base", "0")), ("idea 6 gen nested vs abs nested", ("nest", "x6g"), ("nest", "x6a"))):
            if a[0] == "x4o":
                A = {o: c[o][f"{sysm}|x4o_s1.0"][0] for o in ORDERS}; B = {o: cells(c, o, sysm, "base", "0") for o in ORDERS}; d = {o: A[o] - B[o] for o in ORDERS}; n = c["full0"][f"{sysm}|base"][1]
            else:
                d, n = delta(c, sysm, a, b)
            print(f"{lab:36s} {summarize(d, n, who)}")
        for s in SCALES:
            for form in ("x4c", "x4m", "x4p", "x6g", "x6a"):
                d, n = delta(c, sysm, (form, s), ("base", "0"))
                print(f"  scale {s} {FORMS[form]:30s} {summarize(d, n, who)}")
        for pr in ("uniform", "feas", "feas50", "oracle"):
            if f"{sysm}|prior.{pr}" not in c["full0"]:
                continue
            for lab, ref in (("base", f"{sysm}|base"), ("uniform", f"{sysm}|prior.uniform")):
                if lab == "uniform" and pr == "uniform":
                    continue
                d = {o: c[o][f"{sysm}|prior.{pr}"][0] - c[o][ref][0] for o in ORDERS}
                print(f"idea 5 prior.{pr:8s} vs {lab:8s} {summarize(d, c['full0'][f'{sysm}|base'][1], who)}")


if __name__ == "__main__":
    report(sys.argv[1])
