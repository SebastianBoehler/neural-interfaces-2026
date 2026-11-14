"""RUNDETECT scheme detector (ledger s217, task 1). Per session and delivery order, at window W: ratio = mean distance of
consecutive delivered windows / mean distance of all pairs among the first W, in the leave-one-person-out nuisance
projection (common.projected). Distribution per scheme (recording, runs, within, full; seeds 0-11) and a latched rule
"contiguous runs if ratio < thr", thr chosen on the OTHER people (balanced error: full vs contiguous orders).
Second statistic (added after the first table, s217 result): 'lag', mean distance of pairs <= 3 apart in the delivery /
pairs >= 15 apart; it also sees a boundary inside the prefix (the first one decides 'within', where the plain ratio is
blind until the first run ends). Writes runs/rundetect/ratios.npz ('<stat>|<order>|<session>' -> value per W),
runs/rundetect/decisions.npz (thresholds per held-out person) and prints the tables.
Usage: python studies/14nov/rundetect/detect.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

WS = [20, 30, 40, 50, 60, 80, 100, 150]; STATS = ("ratio", "lag"); SEEDS = range(12); ORDERS = ["recording"] + [f"{sc}{s}" for sc in ("runs", "within", "full") for s in SEEDS]


def stat(X, rows, kind, ws=WS):
    out = []
    for w in ws:
        v = X[rows[:w]]; d = pdist(v); i, j = np.triu_indices(len(v), 1); lag = j - i
        out.append(d[lag == 1].mean() / d.mean() if kind == "ratio" else d[lag <= 3].mean() / d[lag >= 15].mean())
    return np.array(out)


def scheme(o):
    return o.rstrip("0123456789")


def threshold(r, contig):  # midpoint search minimising the balanced error
    cand = np.sort(r); best = (2, 1.0)
    for t in (cand[:-1] + cand[1:]) / 2:
        e = 0.5 * ((r[contig] >= t).mean() + (r[~contig] < t).mean()); best = min(best, (e, t))
    return best[1]


def main():
    X, _ = C.projected(); keys = [(o, s) for o in ORDERS for s in C.SESSIONS]; out, thr_out = {}, {}
    per = np.array([C.PER[C.SES == s][0] for _, s in keys]); sch = np.array([scheme(o) for o, _ in keys]); contig = sch != "full"
    for kind in STATS:
        M = np.stack([stat(X, C.order_rows(o, s), kind) for o, s in keys]); out.update({f"{kind}|{o}|{s}": M[n] for n, (o, s) in enumerate(keys)})
        print(f"RUNDETECT detector (s217) '{kind}': NDIM {C.NDIM} REG {C.REG:g}; quantiles 5/50/95 % over sessions x seeds")
        for sc in ("recording", "runs", "within", "full"):
            m = sch == sc
            print(f"  {sc:9s} " + " | ".join(f"W{w}: {np.percentile(M[m, i], 5):.2f}/{np.median(M[m, i]):.2f}/{np.percentile(M[m, i], 95):.2f}" for i, w in enumerate(WS)))
        print(f"  latched rule 'contiguous if {kind} < thr', thr fitted on the other people (balanced error); error rate per scheme")
        dec = np.zeros(M.shape, bool)
        for p in C.PEOPLE:
            te = per == p; thr_out[f"{kind}|{p}"] = np.array([threshold(M[~te, i], contig[~te]) for i in range(len(WS))]); dec[te] = M[te] < thr_out[f"{kind}|{p}"]
        for sc in ("recording", "runs", "within", "full"):
            m = sch == sc; err = (dec[m] != (sc != "full")).mean(0)
            print(f"  {sc:9s} " + " | ".join(f"W{w}: {e * 100:5.1f} %" for w, e in zip(WS, err)))
        print("  thresholds (median over held-out people): " + " ".join(f"W{w} {np.median([thr_out[f'{kind}|{p}'][i] for p in C.PEOPLE]):.3f}" for i, w in enumerate(WS)))
    np.savez(C.REPO + "/runs/rundetect/ratios.npz", **out); np.savez(C.REPO + "/runs/rundetect/decisions.npz", W=np.array(WS), **thr_out)


if __name__ == "__main__":
    main()
