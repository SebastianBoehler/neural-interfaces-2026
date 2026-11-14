"""Numeric helpers shared by the members and stages."""
import numpy as np
import torch
from scipy.signal import sosfilt, sosfilt_zi

SFREQ = 120.0


def eig_apply(C, fn):
    """fn applied to the eigenvalues of one symmetric matrix."""
    w, V = np.linalg.eigh(C)
    return (V * fn(w)) @ V.T


def eig_apply_batch(C, fn):
    """eig_apply for a stack (B, n, n) of symmetric or Hermitian matrices (torch's batched eigh)."""
    w, V = (t.numpy() for t in torch.linalg.eigh(torch.from_numpy(C)))
    return (V * fn(w)[:, None, :]) @ (V.conj() if np.iscomplexobj(V) else V).transpose(0, 2, 1)


def log_floor(w):
    return np.log(np.maximum(w, 1e-12))


def softmax(z):
    """Softmax over the last axis."""
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def lda_posterior(Z, mu, Ci):
    """Class posteriors of the rows of Z under equal-prior Gaussians with means mu and inverse covariance Ci."""
    W = Ci @ mu.T
    return softmax(Z @ W - 0.5 * (mu * W.T).sum(1))


def filtfilt_setup(sos):
    """(sos, steady state, padding length) for filtfilt: the defaults of scipy's sosfiltfilt, computed once."""
    n = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    return sos, sosfilt_zi(sos), n


def filtfilt(sos, zi, n, x):
    """scipy.signal.sosfiltfilt(sos, x, axis=-1) for 2-D x, without the per-call setup."""
    e = np.concatenate([2 * x[:, :1] - x[:, n:0:-1], x, 2 * x[:, -1:] - x[:, -2:-(n + 2):-1]], -1)
    y = sosfilt(sos, e, zi=zi[:, None] * e[:, :1])[0]
    y = sosfilt(sos, y[:, ::-1], zi=zi[:, None] * y[:, -1:])[0]
    return y[:, ::-1][:, n:-n]
