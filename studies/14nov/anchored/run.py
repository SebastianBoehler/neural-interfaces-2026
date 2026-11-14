"""POPULATION-ANCHORED PERSONAL CLASSIC MODELS (ledger s254): anchored member grids on the 84 W splits of s246.
Labels on the four training runs only: 'true' or the r15 chain's (classic_pl.labels: hard, soft, filt) per order.
Output runs/anchored/{true,<order>}.npz: '<form>|<session>|<pair>|<L|D1|D2>' -> (grid, n_test, 3).
Amendment: 'alpha' -> runs/anchored/alpha_{true,<order>}.npz with the scaled-anchor LR grid only ('...|La').
Usage: python studies/14nov/anchored/run.py check | pop | [alpha] true | [alpha] ORDER [ORDER ...]"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import anchor as A  # noqa: E402
import classic as CL  # noqa: E402
import splits as S  # noqa: E402
from classic_pl import labels  # noqa: E402

POPF = A.OUT / "pop_lr.npz"


def pops(Fs, m):
    """per person: LR population fit from the pop-mode cache, LDA moments and scalers recomputed (fast)"""
    z = np.load(POPF); return {p: A.population(Fs, m, p, (z[p + "|W"], z[p + "|b"])) for p in S.PEOPLE}


def run(Fs, m, P, form, y, w, res, alpha=False):
    for s in S.sessions(m):
        p = s.split("_")[0]
        for name, trn, tst in S.pairs(m, s):
            assert len(np.unique(y[trn])) == 3, (form, s, name)
            for k, v in A.grids(Fs, P[p], y, np.ones(len(trn)) if w is None else w[trn], trn, tst, alpha).items():
                res[f"{form}|{s}|{name}|{k}"] = v.astype(np.float32)


def check(Fs, m, P):
    """pers LR == sklearn LR (same scaler, C 0.1); LDA k 0 == classic.wlda; pop end == population fit; timing"""
    from sklearn.linear_model import LogisticRegression
    y = m["y"]; rng = np.random.default_rng(0)
    for s in ("P01_1", "P04_2"):
        name, trn, tst = S.pairs(m, s)[0]; pp = P[s.split("_")[0]]; w = rng.uniform(0.4, 1, len(trn)); t0 = time.time()
        g = A.grids(Fs, pp, y, w, trn, tst); dt = time.time() - t0; sc = lambda k, X: (X - pp[k + "_sc"][0]) / pp[k + "_sc"][1]  # noqa: E731
        lr = LogisticRegression(C=0.1, max_iter=20000, tol=1e-10).fit(sc("L", Fs["L"][trn]), y[trn], sample_weight=w)
        d1 = np.abs(g["L"][-1] - lr.predict_proba(sc("L", Fs["L"][tst]))).max()
        d2 = max(np.abs(g[k][-1] - CL.pad(*CL.wlda(sc(k, Fs[k][trn]), y[trn], w, sc(k, Fs[k][tst])))).max() for k in ("D1", "D2"))
        d3 = np.abs(g["L"][0] - A.softmax(sc("L", Fs["L"][tst]) @ pp["W"] + pp["b"])).max()
        print(f"{s} {name}: pers LR vs sklearn {d1:.2e}, LDA k0 vs wlda {d2:.2e}, pop end {d3:.1e}, grids {dt:.1f}s", flush=True)
        assert d1 < 1e-4 and d2 < 1e-8 and d3 == 0, (d1, d2, d3)
    print("CHECK OK", flush=True)


def main():
    m = S.meta(); Fs = A.feats(m); arg = sys.argv[1]
    if arg == "pop":
        out = {}
        for p in S.PEOPLE:
            t0 = time.time(); d = A.population(Fs, m, p); out[p + "|W"], out[p + "|b"] = d["W"], d["b"]; print(p, f"{time.time() - t0:.0f}s", flush=True)
        return np.savez(POPF, **out)
    P = pops(Fs, m)
    if arg == "check":
        return check(Fs, m, P)
    PL = np.load(S.OUT / "plabels_r15.npz"); alpha = arg == "alpha"
    for order in sys.argv[1 + alpha:]:
        f = A.OUT / f"{'alpha_' * alpha}{order}.npz"; t0 = time.time(); res = {}
        if f.exists():
            print("exists", f); continue
        if order == "true":
            run(Fs, m, P, "true", m["y"], None, res, alpha)
        else:
            for form in ("hard", "soft", "filt"):
                run(Fs, m, P, form, *labels(PL, order, form), res, alpha)
        np.savez(f, **res); print("saved", f, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
