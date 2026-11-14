"""Tangent-space cores: filter-bank covariances, recentred per session, into a logistic regression."""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from .numerics import SFREQ, eig_apply_batch, filtfilt, filtfilt_setup, log_floor, softmax


class TangentCore:
    """One core. Per window: common-average reference; per band the covariance of the whole window and of ``nsub``
    sub-windows (of the analytic signal when ``hilbert``); each matrix recentred by the session's running
    log-Euclidean mean (the training mean counts as ``k`` windows); tangent vectors plus a slow waveform centred on
    its running mean; standardised multinomial logistic regression."""

    def __init__(self, p):
        self.k, self.shrink, self.nsub = float(p["k"]), float(p["shrink"]), int(p["nsub"])
        self.hilbert = bool(p.get("hilbert", False))
        self.bank = [filtfilt_setup(butter(4, band, btype="band", fs=SFREQ, output="sos")) for band in p["bands"]]
        self.slow_sos = butter(4, float(p["slow_cutoff"]), btype="low", fs=SFREQ, output="sos")
        self.slow_step = int(p["slow_step"])
        self.prior = np.asarray(p["log_glob"])  # (bands * (1 + nsub), C, C): training mean of the matrix logs
        self.mu, self.sd, self.W, self.b = p["mu"], p["sd"], p["W"], p["b"]
        self.reset_state()

    def reset_state(self):
        self.n, self.log_sum, self.slow_sum = 0, np.zeros_like(self.prior), 0.0

    def features(self, x):
        x = x - x.mean(0, keepdims=True)
        self.n += 1
        B = np.stack([filtfilt(*f, x) for f in self.bank]).astype(np.float32)
        if self.hilbert:
            B = hilbert(B, axis=-1).astype(np.complex64)
        C = []
        for seg in [B] + np.array_split(B, self.nsub, axis=-1):
            seg = (seg - seg.mean(-1, keepdims=True)).astype(np.complex128 if self.hilbert else np.float64)
            S = seg @ (seg.conj() if self.hilbert else seg).transpose(0, 2, 1) / seg.shape[-1]
            tr = np.trace(S, axis1=1, axis2=2)[:, None, None].real / S.shape[1]
            C.append((1 - self.shrink) * S + self.shrink * tr * np.eye(S.shape[1]))
        C = np.stack(C, 1).reshape(-1, *C[0].shape[1:])  # band-major: whole window, then the sub-windows
        self.log_sum += eig_apply_batch(C, log_floor)
        ref = (self.k * self.prior + self.log_sum) / (self.k + self.n)
        P = eig_apply_batch(ref, lambda w: np.exp(-0.5 * w))
        T = eig_apply_batch(P @ C @ P, log_floor)
        iu = np.triu_indices(T.shape[-1])
        V = T.real[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
        if self.hilbert:
            ju = np.triu_indices(T.shape[-1], 1)
            V = np.concatenate([V, np.sqrt(2.0) * T.imag[:, ju[0], ju[1]]], 1)
        slow = sosfiltfilt(self.slow_sos, x, axis=-1)[..., self.slow_step // 2::self.slow_step]
        slow = slow.reshape(-1).astype(np.float32).astype(np.float64)
        self.slow_sum = self.slow_sum + slow
        return np.concatenate([V.reshape(-1), slow - self.slow_sum / (self.k + self.n)])

    def step(self, x):
        """Window (C, T) -> (class posterior, standardised features)."""
        z = (self.features(x) - self.mu) / self.sd
        return softmax(self.W @ z + self.b), z
