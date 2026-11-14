"""Shared helpers for the sub-8 Hz view study (ledger s222): band covariances, session references (whole-session for
training sessions, causal with the fold prior for test sessions, as core1), tangent maps, a per-band member and the
replay scorers. Every array is keyed by window index (recording order of data/proteus_windows.npz)."""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/experiments"]
import protocol  # noqa: E402
import features as FT  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

K_PRIOR, SHRINK = 2.0, 0.1
D = protocol.load()
N = len(D["y"])
XCAR = (D["X"] - D["X"].mean(1, keepdims=True)).astype(np.float32)
ST = D["stream"]
SESS = list(dict.fromkeys(ST))
ROWS = [np.flatnonzero(ST == s) for s in SESS]


def band(X, lo, hi):
    return np.concatenate([FT.bandpass(X[i:i + 256], lo, hi) for i in range(0, len(X), 256)])


def cov(Xb, s=SHRINK):
    X = Xb - Xb.mean(-1, keepdims=True); p = X.shape[1]
    S = np.einsum("nct,ndt->ncd", X, X, dtype=np.float64) / X.shape[-1]
    mu = np.trace(S, axis1=1, axis2=2) / p
    return (1 - s) * S + s * mu[:, None, None] * np.eye(p)


def efn(C, fn):
    w, V = np.linalg.eigh(C)
    return np.einsum("...ij,...j,...kj->...ik", V, fn(w), V)


def logm(C):
    return efn(C, lambda w: np.log(np.maximum(w, 1e-12)))


def isqrt(C):
    return efn(C, lambda w: 1 / np.sqrt(np.maximum(w, 1e-12)))


def upper(L):
    n = L.shape[-1]; iu = np.triu_indices(n); w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return L[..., iu[0], iu[1]] * w


def references(C, test_rows, prior_L):
    """Per-window reference R (N, p, p) as log-matrices: whole session for every session, then the test session rows
    replaced by the causal reference (k prior + windows 0..t) / (k + n) in recording order."""
    L = logm(C); R = np.empty_like(L)
    for r in ROWS:
        R[r] = L[r].mean(0)
    for r in test_rows:
        n = np.arange(1, len(r) + 1)[:, None, None]
        R[r] = (K_PRIOR * prior_L + np.cumsum(L[r], 0)) / (K_PRIOR + n)
    return L, R


def tangent(C, Rlog):
    """log(R^-1/2 C R^-1/2), upper triangle; R given as log-matrices."""
    W = efn(Rlog, lambda w: np.exp(-0.5 * w))
    return upper(logm(np.einsum("nij,njk,nkl->nil", W, C, W)))


def session_runs(mask):
    return [r[mask[r]] for r in ROWS if mask[r].any()]


def head(C=0.3):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def lopo(feat_fn, protos=("all_loso", "seen_session"), C=0.3):
    """feat_fn(train_idx, test_mask) -> F (N, d) with training rows whole-session and test rows causal.
    Returns {proto: P (N, 3)} with NaN outside the protocol's test rows."""
    out = {}
    for proto in protos:
        P = np.full((N, 3), np.nan)
        for name, tr, te in protocol.folds(D, proto):
            F = feat_fn(np.flatnonzero(tr), te)
            P[te] = head(C).fit(F[tr], D["y"][tr]).predict_proba(F[te])
        out[proto] = P
    return out


def person_ba(P, proto):
    """cell-mean BA per person through the replay's scorer (rows = replay rows of the protocol)."""
    import env  # noqa: F401
    import score as SC
    idx = SC.B[f"idx|{proto}"].astype(int)
    return SC.person(P[idx], proto)


def summary(Pd):
    a, s = person_ba(Pd["all_loso"], "all_loso"), person_ba(Pd["seen_session"], "seen_session")
    return f"new {a.mean():.2f} seen {s.mean():.2f} | " + " ".join(f"{v:.0f}" for v in a)
