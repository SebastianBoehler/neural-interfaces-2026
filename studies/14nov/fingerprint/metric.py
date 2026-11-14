"""FINGERPRINT METRIC (ledger s250). The s248 hardware view per window (log power 45-58 Hz and 48-52 Hz per channel,
channel-pair correlations at 30-58 Hz; window-local, so causal) in the replay layout, and a run-discriminant metric fitted
on OTHER people's true runs: standardise on the fitting people, PCA, LDA of between-run (run mean minus session mean)
against within-run scatter with shrinkage. The projection is Sw-normalised (within-run SD about 1 per direction).
At use it needs no labels: the held-out person's windows are only projected. Cache: runs/fingerprint/hw.npz."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(__file__ + "/../../rundetect"))
import common as C  # noqa: E402
from scipy.linalg import eigh  # noqa: E402

CACHE = C.REPO + "/runs/fingerprint/hw.npz"
SETTINGS = [(npc, nd, reg) for npc in (60, 120) for nd in (8, 16, 32) for reg in (0.1, 1.0)] + [(200, 64, 0.1), (200, 64, 1.0)]


def features():
    if os.path.exists(CACHE):
        F = np.load(CACHE)["F"]; assert F.shape[0] == C.N; return F.astype(np.float64)
    from scipy.signal import butter, sosfiltfilt
    with np.load(C.REPO + "/data/proteus_windows.npz") as w:
        X = w["X"][C.IDX].astype(np.float64)
    bp = lambda lo, hi: sosfiltfilt(butter(4, [lo, min(hi, 59.4)], "bandpass", fs=C.FS, output="sos"), X, axis=-1)  # noqa: E731
    hb = bp(30, 58); iu = np.triu_indices(X.shape[1], 1)
    cor = np.stack([np.corrcoef(h)[iu] for h in hb])
    F = np.concatenate([np.log(bp(45, 58).var(-1) + 1e-6), np.log(bp(48, 52).var(-1) + 1e-6), cor], 1)
    assert np.isfinite(F).all(); os.makedirs(os.path.dirname(CACHE), exist_ok=True); np.savez(CACHE, F=F.astype(np.float32)); return F


def fit(F, train, setting):
    """train: bool mask of fitting rows (their true runs are used). -> project(Fx) -> (n, nd), lam (nd,)"""
    npc, nd, reg = setting; mu = F[train].mean(0); sd = F[train].std(0) + 1e-9; Zt = (F[train] - mu) / sd
    U, S, Vt = np.linalg.svd(Zt - Zt.mean(0), full_matrices=False); Vp = Vt[:npc].T; Pt = Zt @ Vp
    ses, run = C.SES[train], C.RUN[train]; Pc = Pt.copy()
    for s in np.unique(ses):
        m = ses == s; Pc[m] -= Pt[m].mean(0)
    M = np.zeros_like(Pc)
    for u in np.unique(run):
        m = run == u; M[m] = Pc[m].mean(0)
    Sw = np.cov((Pc - M).T); Sb = np.cov(M.T); Sw += reg * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    lam, V = eigh(Sb, Sw); o = np.argsort(lam)[::-1][:nd]; W = Vp @ V[:, o]
    return (lambda Fx: ((Fx - mu) / sd) @ W), lam[o]


def nn1(Zs, lab):
    D = ((Zs[:, None] - Zs[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf); return (lab[D.argmin(1)] == lab).mean()
