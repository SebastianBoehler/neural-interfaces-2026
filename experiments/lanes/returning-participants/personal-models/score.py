"""Lane G Instrument 1 scorer: arms fitted on training visits only, scored on the held visit under the 8 pinned deliveries.

python score.py OUT ARMS [--out FILE]   (ARMS e.g. stock,g1,g1c,g1a; stock is the control every gain is measured against)
Reads OUT/feats/<person>_<visit>_<order>.npz and OUT/arms/<person>_<visits>_<arm>.npz. Per (split, order, arm, component)
the balanced accuracy of each participant/session/interface cell, equal-cell mean; gains are paired against stock on the
same split and order. Intervals resample the 8 orders (independently per person, same draw for every arm), 80 %.
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as K
from tasks import posteriors

COMPS = ("h", "T", "C", "S", "own", "prod")


def cells(out, arm, person, visits, test, order, kref):
    f = np.load(Path(out) / "feats" / f"{person}_{test}_{order}.npz")
    load = lambda m: dict(np.load(Path(out) / "arms" / f"{person}_{''.join(map(str, visits))}_{m}.npz"))
    if arm == "gS":  # exploratory hybrid (amendment 2): stock arrays, S member from g4
        a, g4 = load("stock"), load("g4")
        a["own_s_W"], a["own_s_c"] = g4["own_s_W"], g4["own_s_c"]
    else:
        a = load(arm)
    a.pop("visits")
    P = posteriors(a, f["u"], f["C"].astype(np.float64), f["sl"], kref)
    return {c: K.ba_cells(P[c].argmax(1), f["y"], f["itf"])[1] for c in COMPS}


def main(out, arms, dest):
    zp = Path(out) / "pkg" / "weights.npz"
    kref = float(np.load(zp)["own_kref"])
    res = {}  # res[arm][comp] -> (splits, orders) array
    for arm in arms:
        res[arm] = {c: np.zeros((len(K.SPLITS), len(K.ORDERS))) for c in COMPS}
        for i, (person, visits, test) in enumerate(K.SPLITS):
            for j, order in enumerate(K.ORDERS):
                r = cells(out, arm, person, visits, test, order, kref)
                for c in COMPS:
                    res[arm][c][i, j] = r[c]
    persons = np.array([s[0] for s in K.SPLITS])
    w = lambda M: 0.2 * M[persons == "P04"].mean(0) + 0.8 * M[persons == "P10"].mean(0)  # per order, 20/80
    rng = np.random.default_rng(20261119)
    draws = [(rng.integers(0, 8, 8), rng.integers(0, 8, 8)) for _ in range(2000)]  # orders per person (P04, P10)

    def boot(M):  # M (splits, orders) -> 80 % interval of the equal-split mean and of the 20/80 mean
        eq, wt = [], []
        for a4, a10 in draws:
            p4, p10 = M[persons == "P04"][:, a4].mean(), M[persons == "P10"][:, a10].mean()
            eq.append((p4 + p10) / 2)
            wt.append(0.2 * p4 + 0.8 * p10)
        return np.percentile(eq, [10, 90]).round(2).tolist(), np.percentile(wt, [10, 90]).round(2).tolist()

    rows, summary = [], {}
    for arm in arms:
        for c in COMPS:
            M = res[arm][c]
            g = M - res["stock"][c]
            p4, p10 = g[persons == "P04"].mean(), g[persons == "P10"].mean()
            eq_i, w_i = boot(g)
            summary[f"{arm}|{c}"] = dict(
                ba=round(float(M.mean()), 2), ba_p04=round(float(M[persons == "P04"].mean()), 2),
                ba_p10=round(float(M[persons == "P10"].mean()), 2), gain_equal=round(float((p4 + p10) / 2), 2),
                gain_2080=round(float(0.2 * p4 + 0.8 * p10), 2), gain_p04=round(float(p4), 2), gain_p10=round(float(p10), 2),
                orders_nonneg_2080=int((w(g) >= 0).sum()), ci80_equal=eq_i, ci80_2080=w_i,
                per_split=[round(float(x), 2) for x in g.mean(1)])
            rows.append(f"| {arm} | {c} | {M.mean():.2f} | {summary[f'{arm}|{c}']['gain_equal']:+.2f} {eq_i} | "
                        f"{summary[f'{arm}|{c}']['gain_2080']:+.2f} {w_i} | {p4:+.2f} / {p10:+.2f} | "
                        f"{summary[f'{arm}|{c}']['orders_nonneg_2080']}/8 |")
    print("| arm | component | BA | gain equal [80%] | gain 20/80 [80%] | P04 / P10 | orders >=0 |\n|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    Path(dest).write_text(json.dumps(dict(summary=summary, splits=[list(map(str, s)) for s in K.SPLITS], orders=K.ORDERS,
                                          raw={a: {c: res[a][c].tolist() for c in COMPS} for a in arms}), indent=1))


if __name__ == "__main__":
    out, arms = sys.argv[1], sys.argv[2].split(",")
    assert arms[0] == "stock", "stock first: it is the paired control"
    main(out, arms, sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else str(Path(out) / "score.json"))
