"""CAUSAL PERSONALISATION run time (ledger s253), Mac CPU, one process, warm (each timing is the median of REP repeats).
Per pipeline: per-window features (the bands it needs, one 41 x 480 window), refit on n windows < t (n 60 / 120 / 180, soft
weights, pseudo-labels from the chain), predict one window. Per window at refit interval k: feat + predict + refit(n) / k.
Usage: python studies/14nov/causalpers/timing.py"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path[:0] = [str(HERE), str(REPO / "studies/14nov/personal_ceiling")]
import numpy as np  # noqa: E402
from pyriemann.classification import MDM  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import classic as CL  # noqa: E402
import feats as FT  # noqa: E402

REP = 5; NEED = {"a1": CL.TSB, "a2": CL.CSPB, "a3": [(8, 30)], "a4": []}


def med(f):
    ts = []
    for _ in range(REP):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def fits(F, y, w, tr, te):
    """component refits mirroring classic.fit_predict -> {pipe: (fit fn, predict fn)}"""
    st = {}

    def f1():
        st["ts"] = [TangentSpace("riemann").fit(F[f"cov{lo}_{hi}"][tr]) for lo, hi in CL.TSB]
        Z = np.concatenate([t.transform(F[f"cov{lo}_{hi}"][tr]) for t, (lo, hi) in zip(st["ts"], CL.TSB)], 1)
        st["lr"] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Z, y[tr], logisticregression__sample_weight=w[tr])

    def p1():
        st["lr"].predict_proba(np.concatenate([t.transform(F[f"cov{lo}_{hi}"][te]) for t, (lo, hi) in zip(st["ts"], CL.TSB)], 1))

    def f2():
        pr = [CL.csp_feats(F[f"cov{lo}_{hi}"][tr], y[tr], F[f"cov{lo}_{hi}"][te], w=w[tr]) for lo, hi in CL.CSPB]
        CL.wlda(np.concatenate([a for a, _ in pr], 1), y[tr], w[tr], np.concatenate([b for _, b in pr], 1))

    def f3():
        st["md"] = MDM(metric="riemann").fit(F["cov8_30"][tr], y[tr], sample_weight=w[tr])

    def f4():
        CL.wlda(F["slow"][tr], y[tr], w[tr], F["slow"][te])
    return {"a1": (f1, p1), "a2": (f2, lambda: None), "a3": (f3, lambda: st["md"].predict_proba(F["cov8_30"][te])), "a4": (f4, lambda: None)}


def main():
    F = FT.load(); Z = np.load(REPO / "runs/causalpers/chain.npz"); P = Z["full0|0|filt"]; y, w = P.argmax(1), P.max(1)
    with np.load(REPO / "data/proteus_windows.npz") as d:
        x1 = d["X"][:1].astype(np.float64)
    rows = Z["full0|deliv"][:200]; feat = {p: med(lambda: FT.window_feats(x1, NEED[p])) if NEED[p] else med(lambda: FT.window_feats(x1, [])) for p in NEED}
    print("per-window features (ms): " + " ".join(f"{p} {v:.1f}" for p, v in feat.items()) + f" | all {med(lambda: FT.window_feats(x1)):.1f}")
    res = {}
    for n in (60, 120, 180):
        tr, te = rows[:n], rows[n:n + 1]; fs = fits(F, y, w, tr, te)
        for p, (f, pr) in fs.items():
            ft = med(f); pt = med(pr) if p in ("a1", "a3") else 0.0; res[p, n] = (ft, pt)  # a2 / a4 predict is inside their fit call (tiny)
        print(f"n {n}: refit ms " + " ".join(f"{p} {res[p, n][0]:.0f}" for p in fs) + " | predict ms " + " ".join(f"{p} {res[p, n][1]:.1f}" for p in fs), flush=True)
    print("per window (ms), refit at n 120 amortised over k:")
    for k in (5, 10, 20):
        a = {p: feat[p] + res[p, 120][1] + res[p, 120][0] / k for p in NEED}; aE = sum(a.values()) - sum(feat.values()) + FT_all if (FT_all := med(lambda: FT.window_feats(x1))) else 0
        print(f"  k {k}: " + " ".join(f"{p} {v:.1f}" for p, v in a.items()) + f" | aE {aE:.1f}")


if __name__ == "__main__":
    main()
