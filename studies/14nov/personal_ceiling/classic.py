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


def csp_feats(Ctr, ytr, Cte, nf=2):
    """one-vs-rest CSP on band covariances -> log-variance features (train, test)"""
    W = []
    for c in np.unique(ytr):
        A, B = Ctr[ytr == c].mean(0), Ctr[ytr != c].mean(0); B = 0.9 * B + 0.1 * np.trace(B) / len(B) * np.eye(len(B))
        _, V = eigh(A, A + B); W += [V[:, :nf], V[:, -nf:]]
    W = np.concatenate(W, 1); lv = lambda C: np.log(np.einsum("ck,ncd,dk->nk", W, C, W))  # noqa: E731
    return lv(Ctr), lv(Cte)


def fit_predict(F, y, tr, te):
    """-> {variant: (len(te), 3)}"""
    out = {}
    ts = [TangentSpace("riemann").fit(F[f"cov{lo}_{hi}"][tr]) for lo, hi in TSB]
    Ztr = np.concatenate([t.transform(F[f"cov{lo}_{hi}"][tr]) for t, (lo, hi) in zip(ts, TSB)], 1)
    Zte = np.concatenate([t.transform(F[f"cov{lo}_{hi}"][te]) for t, (lo, hi) in zip(ts, TSB)], 1)
    out["a1"] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Ztr, y[tr]).predict_proba(Zte)
    pr = [csp_feats(F[f"cov{lo}_{hi}"][tr], y[tr], F[f"cov{lo}_{hi}"][te]) for lo, hi in CSPB]
    out["a2"] = LDA(solver="lsqr", shrinkage="auto").fit(np.concatenate([a for a, _ in pr], 1), y[tr]).predict_proba(np.concatenate([b for _, b in pr], 1))
    out["a3"] = MDM(metric="riemann").fit(F["cov8_30"][tr], y[tr]).predict_proba(F["cov8_30"][te])
    out["a4"] = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")).fit(F["slow"][tr], y[tr]).predict_proba(F["slow"][te])
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
