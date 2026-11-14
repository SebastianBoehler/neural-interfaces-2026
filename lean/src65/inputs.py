"""Input handling: montage mapping, sanitising, and the causal per-session network input."""
import numpy as np
import torch

from .numerics import eig_apply

# Training montage order (PROTEUS meta["ch_names"])
CH_NAMES = ("C2", "CP2", "P2", "Pz", "CPz", "Cz", "C1", "CP1", "P1", "P3", "PO7", "P5", "TP7",
            "CP5", "CP3", "C3", "PO8", "P6", "P4", "TP8", "CP6", "CP4", "C6", "C4", "FC6", "FC4",
            "FC2", "F6", "F4", "F2", "AF4", "AF3", "F5", "F3", "F1", "Fz", "FC5", "FC3", "FC1",
            "FCz", "C5")


def channel_map(names):
    """Index of each training channel in the runtime montage (-1: missing), or None when the runtime order is the
    training order or the names are unusable (inputs then stay positional)."""
    try:
        names = [str(c).strip().upper() for c in names]
    except TypeError:
        return None
    train = [c.upper() for c in CH_NAMES]
    if names == train or len(set(names) & set(train)) < len(train) // 2:
        return None
    return np.array([names.index(c) if c in names else -1 for c in train])


def to_windows(X, perm):
    """Harness batch (B, C, T) -> finite float32 array in training channel order (missing channels zero)."""
    X = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
    if perm is not None:
        Xn = np.zeros((len(X), len(perm), X.shape[-1]), np.float32)
        Xn[:, perm >= 0] = X[:, perm[perm >= 0]]
        X = Xn
    return np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


INPUT_KEYS = ("align", "k", "G", "scale", "shrink")  # per-net input parameters


def input_key(p):
    """Nets with equal input parameters share one InputStream."""
    return b"".join(np.ascontiguousarray(p[k]).tobytes() for k in INPUT_KEYS)


class InputStream:
    """Network input of one window: divided by a global scale, or (``align``) whitened by the session's running
    mean covariance, the training mean G counting as ``k`` windows."""

    def __init__(self, p):
        self.align, self.scale, self.G = bool(p["align"]), float(p["scale"]), p["G"]
        self.k, self.shrink = float(p["k"]), float(p["shrink"])
        self.reset_state()

    def reset_state(self):
        self.n, self.cov_sum = 0, 0.0

    def __call__(self, x):
        X = x[None].astype(np.float64)
        if not self.align:
            return (X / self.scale).astype(np.float32)[0]
        xc = x - x.mean(-1, keepdims=True)
        C = np.einsum("nct,ndt->ncd", xc[None], xc[None], dtype=np.float64) / x.shape[-1]
        tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
        C = (1 - self.shrink) * C + self.shrink * tr * np.eye(C.shape[1])
        self.n += 1
        self.cov_sum = self.cov_sum + C[0]
        R = (self.k * self.G + self.cov_sum) / (self.k + self.n)
        P = eig_apply(R, lambda w: 1.0 / np.sqrt(np.maximum(w, 1e-12)))
        return np.einsum("ij,njt->nit", P, X).astype(np.float32)[0]
