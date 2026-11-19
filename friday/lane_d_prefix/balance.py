"""Label-free probability transformations shared by Lane D replays and package."""

import numpy as np


def terminal_probability(values):
    """Before CSP fusion cap250 returns probabilities, afterwards it returns log scores."""
    p = np.array(values, dtype=np.float64, copy=True)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("invalid cap250 terminal scores")
    fused = (p < 0).any(1)
    if np.any(p[fused] > 0) or not np.allclose(p[~fused].sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError("unexpected cap250 score representation")
    logs = p[fused]
    p[fused] = np.exp(logs - logs.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def balanced_probabilities(history):
    """KL projection with unit row mass and equal column mass."""
    p = np.maximum(np.asarray(history, dtype=np.float64), 1e-12)
    scale = np.ones(3)
    for _ in range(512):
        q = p * scale
        q /= q.sum(1, keepdims=True)
        mass = q.mean(0)
        if np.max(np.abs(mass - 1 / 3)) <= 1e-8:
            return q
        scale /= 3 * mass
        scale /= scale.max()
    raise RuntimeError("Lane D Sinkhorn did not converge")


class PrefixBalance:
    def __init__(self):
        self.history = []

    def step(self, raw):
        p = terminal_probability(np.asarray(raw)[None])[0]
        self.history.append(p)
        if len(self.history) < 20:
            return raw
        return balanced_probabilities(self.history)[-1]
