"""PERSONAL CEILING (a) (ledger s246): classic personal pipelines, trained only on the person's own labelled runs.
Every session is Euclidean-aligned on its own whole-session mean covariance (non-causal, unlabelled; same rule for the
training and the test session, W and X). Features, all derived from per-window band covariances (OAS-shrunk):
  a1  filter-bank tangent space (1-4, 4-8, 8-13, 13-20, 20-30, 30-45 Hz; reference = training mean) + L2 LR (C 0.1)
  a2  FBCSP: per band (4-8 ... 32-40, eight bands) one-vs-rest CSP, 2 + 2 filters per class, log-variance, shrinkage LDA
  a3  MDM on the 8-30 Hz covariance
  a4  slow waveform: < 4 Hz low-pass, every 15th sample (8 Hz, 32 per channel), shrinkage LDA
  aE  mean of the four probabilities
Output runs/personal_ceiling/classic.npz in ftnet.py's layout ('<W|X>|<session>|<pair>|<variant>', '...|idx').
Usage: python studies/14nov/personal_ceiling/classic.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
from pyriemann.classification import MDM  # noqa: E402
from pyriemann.estimation import Covariances  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import splits as S  # noqa: E402

FS = 120.0; TSB = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45)]; CSPB = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 40)]


def isqrt(C):
    w, V = np.linalg.eigh(C); return (V / np.sqrt(w)) @ V.T


def features(m):
    """EA per session, band covariances for every band used, slow waveform."""
    f = S.OUT / "classic_feat.npz"
    if f.exists():
        z = np.load(f); return {k: z[k] for k in z.files}
    with np.load(S.REPO / "data/proteus_windows.npz") as w:
        X = w["X"].astype(np.float64)
    for s in S.sessions(m):
        r = m["sess"] == s; C = np.einsum("nct,ndt->cd", X[r], X[r]) / (r.sum() * X.shape[-1]); X[r] = np.einsum("ij,njt->nit", isqrt(C), X[r])
    out = {}
    for lo, hi in sorted(set(TSB + CSPB + [(8, 30)])):
        Xb = sosfiltfilt(butter(4, (lo, hi), "bandpass", fs=FS, output="sos"), X, axis=-1); out[f"cov{lo}_{hi}"] = Covariances("oas").fit_transform(Xb)
    out["slow"] = sosfiltfilt(butter(4, 4.0, "lowpass", fs=FS, output="sos"), X, axis=-1)[:, :, ::15].reshape(len(X), -1)
    np.savez(f, **out); return out


def csp_feats(Ctr, ytr, Cte, nf=2, w=None):
    """one-vs-rest CSP on band covariances (class means weighted by w) -> log-variance features (train, test)"""
    w = np.ones(len(ytr)) if w is None else w; W = []
    for c in np.unique(ytr):
        a, b = ytr == c, ytr != c; A, B = np.average(Ctr[a], 0, w[a]), np.average(Ctr[b], 0, w[b]); B = 0.9 * B + 0.1 * np.trace(B) / len(B) * np.eye(len(B))
        _, V = eigh(A, A + B); W += [V[:, :nf], V[:, -nf:]]
    W = np.concatenate(W, 1); lv = lambda C: np.log(np.einsum("ck,ncd,dk->nk", W, C, W))  # noqa: E731
    return lv(Ctr), lv(Cte)


def wlda(Xtr, y, w, Xte):
    """shrinkage LDA with sample weights (s252 soft forms): weighted class means and priors, Ledoit-Wolf on the weighted
    within-class residuals"""
    from sklearn.covariance import ledoit_wolf
    cl = np.unique(y); mu = np.stack([np.average(Xtr[y == c], 0, w[y == c]) for c in cl]); R = (Xtr - mu[np.searchsorted(cl, y)]) * np.sqrt(w / w.mean())[:, None]
    Sg = ledoit_wolf(R, assume_centered=True)[0]; B = np.linalg.solve(Sg, mu.T); pri = np.array([w[y == c].sum() for c in cl]) / w.sum()
    L = Xte @ B - 0.5 * np.sum(mu.T * B, 0) + np.log(pri); L -= L.max(1, keepdims=True); E = np.exp(L); return cl, E / E.sum(1, keepdims=True)


def pad(cl, P):
    """-> (n, 3) with columns for classes absent from the training labels at 0"""
    out = np.zeros((len(P), 3)); out[:, np.asarray(cl, int)] = P; return out


def fit_predict(F, y, tr, te, w=None):
    """-> {variant: (len(te), 3)}; y labels used on tr only (true or pseudo); w sample weights (None: unweighted, sklearn
    LDA as in s246; else weighted LR / MDM / CSP means and wlda)"""
    out = {}; yt = y[tr]; ts = [TangentSpace("riemann").fit(F[f"cov{lo}_{hi}"][tr]) for lo, hi in TSB]
    Ztr = np.concatenate([t.transform(F[f"cov{lo}_{hi}"][tr]) for t, (lo, hi) in zip(ts, TSB)], 1)
    Zte = np.concatenate([t.transform(F[f"cov{lo}_{hi}"][te]) for t, (lo, hi) in zip(ts, TSB)], 1)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Ztr, yt, logisticregression__sample_weight=w)
    out["a1"] = pad(lr.classes_, lr.predict_proba(Zte))
    pr = [csp_feats(F[f"cov{lo}_{hi}"][tr], yt, F[f"cov{lo}_{hi}"][te], w=w) for lo, hi in CSPB]; A2, B2 = np.concatenate([a for a, _ in pr], 1), np.concatenate([b for _, b in pr], 1)
    sc = StandardScaler().fit(F["slow"][tr]); A4, B4 = sc.transform(F["slow"][tr]), sc.transform(F["slow"][te])
    if w is None:
        l2 = LDA(solver="lsqr", shrinkage="auto").fit(A2, yt); l4 = LDA(solver="lsqr", shrinkage="auto").fit(A4, yt)
        out["a2"], out["a4"] = pad(l2.classes_, l2.predict_proba(B2)), pad(l4.classes_, l4.predict_proba(B4))
    else:
        out["a2"], out["a4"] = pad(*wlda(A2, yt, w, B2)), pad(*wlda(A4, yt, w, B4))
    md = MDM(metric="riemann").fit(F["cov8_30"][tr], yt, sample_weight=w); out["a3"] = pad(md.classes_, md.predict_proba(F["cov8_30"][te]))
    out["aE"] = np.mean([out[k] for k in ("a1", "a2", "a3", "a4")], 0)
    for k, v in out.items():
        assert v.shape == (len(te), 3) and np.isfinite(v).all(), k
    return out


def main():
    t0 = time.time(); m = S.meta(); F = features(m); y = m["y"]; res = {}
    print(f"features {time.time() - t0:.0f}s", flush=True)
    for s in S.sessions(m):
        for name, trn, tst in S.pairs(m, s):
            res[f"W|{s}|{name}|idx"] = tst
            for k, v in fit_predict(F, y, trn, tst).items():
                res[f"W|{s}|{name}|{k}"] = v
        print(s, " ".join(f"{k} {np.mean([S.cell_score(m, res[f'W|{s}|{n}|idx'], res[f'W|{s}|{n}|{k}']) for n, _, _ in S.pairs(m, s)]):.1f}" for k in ("a1", "a2", "a3", "a4", "aE")), f"{time.time() - t0:.0f}s", flush=True)
    for p, s_tr, s_te in S.cross(m):
        r1 = np.flatnonzero(m["sess"] == s_tr)
        for name, _, tst in S.pairs(m, s_te) + [("all", None, np.flatnonzero(m["sess"] == s_te))]:
            res[f"X|{s_te}|{name}|idx"] = tst
            for k, v in fit_predict(F, y, r1, tst).items():
                res[f"X|{s_te}|{name}|{k}"] = v
        print("X", s_tr, "->", s_te, " ".join(f"{k} {S.cell_score(m, res[f'X|{s_te}|all|idx'], res[f'X|{s_te}|all|{k}']):.1f}" for k in ("a1", "a2", "a3", "a4", "aE")), flush=True)
    np.savez(S.OUT / "classic.npz", **res); print("saved", S.OUT / "classic.npz", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
