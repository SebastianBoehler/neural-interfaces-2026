"""Per-window features and SPD helpers (each window is processed alone)."""

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

SFREQ = 120.0
CACHE = Path(__file__).resolve().parents[1] / "data/feature_cache"
BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta1": (13, 20), "beta2": (20, 30),
    "gamma": (30, 45), "high": (45, 58), "broad": (1, 45),
}


def bandpass(X, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=SFREQ, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def slow_waveform(X, cutoff=2.0, step=60):
    """Low-pass waveform sampled every ``step`` samples, (N, C*T')."""
    sos = butter(4, cutoff, btype="low", fs=SFREQ, output="sos")
    Y = sosfiltfilt(sos, X, axis=-1)[..., step // 2::step]
    return Y.reshape(len(X), -1).astype(np.float32)


def covariances(X, shrink=1e-3):
    """Trace-shrunk sample covariances, (N, C, T) -> (N, C, C) float64."""
    X = X - X.mean(-1, keepdims=True)
    C = np.einsum("nct,ndt->ncd", X, X, dtype=np.float64) / X.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
    eye = np.eye(C.shape[1])
    return (1 - shrink) * C + shrink * tr * eye


def band_covariances(X, band, tag=""):
    """Cached covariances of all windows for one band (``tag``: external data)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (f"cov_{band}@{tag}.npy" if tag else f"cov_{band}.npy")
    if path.exists():
        return np.load(path)
    lo, hi = BANDS[band]
    C = np.concatenate([covariances(bandpass(X[i:i + 256], lo, hi))
                        for i in range(0, len(X), 256)])
    np.save(path, C)
    return C


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return np.einsum("...ij,...j,...kj->...ik", V, fn(w), V)


def logm(C):
    return _eig_fn(C, lambda w: np.log(np.maximum(w, 1e-12)))


def expm_sym(L):
    return _eig_fn(L, np.exp)


def invsqrtm(C):
    return _eig_fn(C, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-12)))


def upper(S):
    """Tangent vector of symmetric matrices with sqrt(2) off-diagonal weight."""
    n = S.shape[-1]
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return S[..., iu[0], iu[1]] * coef


def causal_references(logC, log_prior, k):
    """Log-Euclidean running means: window t uses windows 0..t and a prior.

    ``logC`` (T, C, C) are matrix logs of one session's covariances in
    acquisition order; ``log_prior`` is the log of a training-set reference
    weighted as ``k`` pseudo-windows. Returns references (T, C, C).
    """
    csum = np.cumsum(logC, axis=0)
    n = np.arange(1, len(logC) + 1)[:, None, None]
    return expm_sym((k * log_prior + csum) / (k + n))


def whiten_tangent(C, R):
    """Tangent vectors of C after recentering by reference(s) R at identity."""
    P = invsqrtm(R)
    W = P @ C @ P if P.ndim == 2 else np.einsum("nij,njk,nlk->nil", P, C, P)
    return upper(logm(W))
