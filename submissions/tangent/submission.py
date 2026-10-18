"""Session-aligned filter-bank tangent-space decoder (Track 2 submission).

Each window is band-pass filtered into several bands; each band's shrunk
covariance is recentred by a causal reference of the current session (the
log-Euclidean mean of this and earlier windows, blended with a training
reference as ``k`` pseudo-windows) and mapped to the tangent space. Slow
features (channel means, or a low-pass waveform sampled every ``slow_step``
samples), centred by their causal session mean, are appended. A standardised
linear classifier predicts the command. State resets at every session and
uses no labels. Parameters are read from ``weights.npz``.
"""

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from benchmark_utils.base_solver import CompetSolver

SFREQ = 120.0


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)) @ V.T


def _log(C):
    return _eig_fn(C, lambda w: np.log(np.maximum(w, 1e-12)))


def _upper(S):
    iu = np.triu_indices(S.shape[-1])
    return S[iu] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


class AlignedTangent:
    def __init__(self, params):
        self.p = params
        self.sos = [butter(4, [lo, hi], btype="band", fs=SFREQ, output="sos")
                    for lo, hi in params["bands"]]
        cut = float(params.get("slow_cutoff", 0.0))
        self.slow_sos = (butter(4, cut, btype="low", fs=SFREQ, output="sos")
                         if cut > 0 else None)
        self.reset_state()

    def _slow(self, x):
        if self.slow_sos is None:
            return x.mean(-1)
        step = int(self.p["slow_step"])
        y = sosfiltfilt(self.slow_sos, x, axis=-1)[..., step // 2::step]
        return y.reshape(-1).astype(np.float32)

    def reset_state(self):
        self.n = 0
        self.log_sum = [np.zeros_like(g) for g in self.p["log_glob"]]
        self.mean_sum = 0.0

    def _cov(self, x):
        x = x - x.mean(-1, keepdims=True)
        C = (x.astype(np.float64) @ x.T.astype(np.float64)) / x.shape[-1]
        s = float(self.p["shrink"])
        return (1 - s) * C + s * np.trace(C) / len(C) * np.eye(len(C))

    def _features(self, x):
        self.n += 1
        k = float(self.p["k"])
        feats = []
        for b, sos in enumerate(self.sos):
            C = self._cov(sosfiltfilt(sos, x, axis=-1).astype(np.float32))
            L = _log(C)
            self.log_sum[b] += L
            ref = (k * self.p["log_glob"][b] + self.log_sum[b]) / (k + self.n)
            P = _eig_fn(ref, lambda w: np.exp(-0.5 * w))  # expm(ref) ** -1/2
            feats.append(_upper(_log(P @ C @ P)))
        m = self._slow(x)
        self.mean_sum = self.mean_sum + m  # float32, like the training cumsum
        feats.append(m - self.mean_sum / (k + self.n))
        return np.concatenate(feats)

    @torch.no_grad()
    def predict(self, X):
        X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
        out = []
        for x in X.astype(np.float32):
            f = (self._features(x) - self.p["mu"]) / self.p["sd"]
            out.append(int(np.argmax(self.p["W"] @ f + self.p["b"])))
        return torch.as_tensor(out, dtype=torch.long)


class Solver(CompetSolver):

    name = "AlignedTangent"

    def load_model(self, meta):
        with np.load(meta["submission_dir"] / "weights.npz") as f:
            params = {k: f[k] for k in f.files}
        params["log_glob"] = list(params["log_glob"])
        params["bands"] = [tuple(b) for b in params["bands"]]
        return AlignedTangent(params)
