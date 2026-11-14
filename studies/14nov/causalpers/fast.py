"""CAUSAL PERSONALISATION fast pipelines (ledger s253 amendment). The Riemannian mean inside classic.py's TangentSpace / MDM
is ~90 % of a refit (eigh loops). Here the tangent space is log-Euclidean: per window, once, the matrix log of each band
covariance (upper triangle, off-diagonal x sqrt 2); the reference subtraction is absorbed by the StandardScaler, so a refit
is the LR alone. MDM (a3, 20 points below the chain in s246) is dropped.
  l1  log-Euclidean FB tangent space (classic.TSB) + StandardScaler + L2 LR (C 0.1), sample weights
  a2  classic FBCSP (classic.csp_feats) + weighted shrinkage LDA (classic.wlda) / sklearn LDA when unweighted
  a4  classic slow waveform + the same LDA
  lE  mean of l1, a2, a4
Usage: python studies/14nov/causalpers/fast.py check   (true-label W splits of s246: fast vs classic, no EA)"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]; sys.path[:0] = [str(HERE), str(REPO / "studies/14nov/personal_ceiling")]
import numpy as np  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import classic as CL  # noqa: E402

PIPES = ("l1", "a2", "a4", "lE")


def logvec(C):
    """(n, c, c) SPD -> (n, c(c+1)/2) upper triangle of logm, off-diagonal x sqrt 2 (per window, no state)"""
    w, V = np.linalg.eigh(C); L = np.einsum("nij,nj,nkj->nik", V, np.log(w), V); iu = np.triu_indices(C.shape[1])
    return L[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2))


def tsle(F):
    return np.concatenate([logvec(F[f"cov{lo}_{hi}"]) for lo, hi in CL.TSB], 1)


def lda(A, y, w, B):
    if w is None:
        m = LDA(solver="lsqr", shrinkage="auto").fit(A, y); return CL.pad(m.classes_, m.predict_proba(B))
    return CL.pad(*CL.wlda(A, y, w, B))


def fit_predict(F, T, y, tr, te, w=None):
    """F classic-keyed per-window features, T = tsle(F); y labels used on tr only -> {pipe: (len(te), 3)}"""
    yt = y[tr]; out = {}
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(T[tr], yt, logisticregression__sample_weight=w)
    out["l1"] = CL.pad(lr.classes_, lr.predict_proba(T[te]))
    pr = [CL.csp_feats(F[f"cov{lo}_{hi}"][tr], yt, F[f"cov{lo}_{hi}"][te], w=w) for lo, hi in CL.CSPB]
    out["a2"] = lda(np.concatenate([a for a, _ in pr], 1), yt, w, np.concatenate([b for _, b in pr], 1))
    sc = StandardScaler().fit(F["slow"][tr]); out["a4"] = lda(sc.transform(F["slow"][tr]), yt, w, sc.transform(F["slow"][te]))
    out["lE"] = np.mean([out[k] for k in ("l1", "a2", "a4")], 0)
    for k, v in out.items():
        assert v.shape == (len(te), 3) and np.isfinite(v).all(), k
    return out


def check():
    import time
    import feats as FT
    import splits as S
    m = S.meta(); F = FT.load(); T = tsle(F); y = m["y"]; ref = np.load(S.OUT / "classic.npz"); acc = {}; t0 = time.time()
    for s in S.sessions(m):
        for name, trn, tst in S.pairs(m, s):
            got = fit_predict(F, T, y, trn, tst); assert (ref[f"W|{s}|{name}|idx"] == tst).all()
            for k, v in got.items():
                acc.setdefault(k, []).append(S.cell_score(m, tst, v))
            for k in ("a1", "a2", "a4", "aE"):
                acc.setdefault("s246 EA " + k, []).append(S.cell_score(m, tst, ref[f"W|{s}|{name}|{k}"]))
    print(f"true-label W splits (84), mean over splits; {time.time() - t0:.0f}s")
    for k, v in acc.items():
        print(f"  {k:12s} {np.mean(v):.2f}")


if __name__ == "__main__":
    check()
