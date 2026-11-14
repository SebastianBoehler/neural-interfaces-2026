"""CAUSAL PERSONALISATION run time (ledger s253), Mac CPU, one process, warm (median of REP repeats). Per pipeline of
fast.py: per-window features for one 41 x 480 window (the bands it needs; l1 adds the matrix logs), refit on n windows < t
(n 60 / 120 / 180, soft weights from the chain), predict one window. Per window at refit interval k: feat + predict +
refit(120) / k.
Usage: [OMP_NUM_THREADS=1] python studies/14nov/causalpers/timing.py"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path[:0] = [str(HERE), str(REPO / "studies/14nov/personal_ceiling")]
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import classic as CL  # noqa: E402
import fast as FA  # noqa: E402
import feats as FT  # noqa: E402

REP = 7


def med(f):
    f(); ts = []
    for _ in range(REP):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def main():
    F = FT.load(); T = FA.tsle(F); Z = np.load(REPO / "runs/causalpers/chain.npz"); P = Z["full0|0|filt"]; y, w = P.argmax(1), P.max(1)
    with np.load(REPO / "data/proteus_windows.npz") as d:
        x1 = d["X"][:1].astype(np.float64)
    nb = lambda B: lambda: FT.window_feats(x1, B)  # noqa: E731
    feat = {"l1": med(lambda: FA.tsle(FT.window_feats(x1, CL.TSB))), "a2": med(nb(CL.CSPB)), "a4": med(nb([]))}
    feat["lE"] = med(lambda: FA.tsle(FT.window_feats(x1, sorted(set(CL.TSB + CL.CSPB)))))
    print("per-window features (ms): " + " ".join(f"{p} {v:.1f}" for p, v in feat.items()))
    rows = Z["full0|deliv"][:200]; fit, pred = {}, {}
    for n in (60, 120, 180):
        tr, te = rows[:n], rows[n:n + 1]; st = {}

        def f1():
            st["lr"] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(T[tr], y[tr], logisticregression__sample_weight=w[tr])

        def f2():
            pr = [CL.csp_feats(F[f"cov{lo}_{hi}"][tr], y[tr], F[f"cov{lo}_{hi}"][te], w=w[tr]) for lo, hi in CL.CSPB]
            CL.wlda(np.concatenate([a for a, _ in pr], 1), y[tr], w[tr], np.concatenate([b for _, b in pr], 1))

        def f4():
            sc = StandardScaler().fit(F["slow"][tr]); CL.wlda(sc.transform(F["slow"][tr]), y[tr], w[tr], sc.transform(F["slow"][te]))
        fit["l1", n], fit["a2", n], fit["a4", n] = med(f1), med(f2), med(f4); pred["l1"] = med(lambda: st["lr"].predict_proba(T[te]))
        fit["lE", n] = fit["l1", n] + fit["a2", n] + fit["a4", n]
        print(f"n {n}: refit ms " + " ".join(f"{p} {fit[p, n]:.0f}" for p in ("l1", "a2", "a4", "lE")) + f" | l1 predict {pred['l1']:.2f} ms (a2 / a4 predict inside the refit, < 1 ms)", flush=True)
    print("per window (ms) = features + predict + refit(n 120) / k:")
    for k in (10, 20):
        print(f"  k {k}: " + " ".join(f"{p} {feat[p] + pred['l1'] + fit[p, 120] / k:.1f}" for p in ("l1", "a2", "a4", "lE")))


if __name__ == "__main__":
    main()
