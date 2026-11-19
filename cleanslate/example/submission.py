"""Clean-slate example team: causal Euclidean alignment + tangent space + logistic regression (smoke system).

State (running covariance sum of this session's delivered windows) resets per session; nothing crosses sessions.
Weights: weights.npz (standardiser mean/scale, LR coef/intercept).  Training reuses `Aligner` from this very file.
"""
import numpy as np
import torch

from benchmark_utils.base_solver import CompetSolver

N0 = 10.0  # pseudo-windows of identity shrinkage for the first windows of a session


def _eigfun(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)) @ V.T


class Aligner:
    """Causal EA: whiten each window's covariance by the mean covariance of this and earlier windows."""

    def __init__(self, d=41):
        self.d = d
        self.reset()

    def reset(self):
        self.S, self.n = np.zeros((self.d, self.d)), 0

    def step(self, x):
        x = x.astype(np.float64)
        x = x - x.mean(1, keepdims=True)
        C = x @ x.T / x.shape[1]
        self.S += C
        self.n += 1
        R = self.S / self.n
        R = (self.n * R + N0 * np.trace(R) / self.d * np.eye(self.d)) / (self.n + N0)
        W = _eigfun(R, lambda w: 1 / np.sqrt(np.maximum(w, 1e-10)))
        Z = _eigfun(W @ C @ W + 1e-6 * np.eye(self.d), lambda w: np.log(np.maximum(w, 1e-12)))
        iu = np.triu_indices(self.d)
        return Z[iu] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


class _Model:
    def __init__(self, w, device):
        self.w, self.al = w, Aligner()

    def reset_state(self):
        self.al.reset()

    def predict(self, X):
        out = []
        for x in X.detach().cpu().numpy():
            f = (self.al.step(x) - self.w["mu"]) / self.w["sd"]
            out.append(int(np.argmax(self.w["coef"] @ f + self.w["b"])))
        return torch.tensor(out, dtype=torch.int64)


class Solver(CompetSolver):
    name = "cs_example"

    def load_model(self, meta):
        with np.load(f"{meta['submission_dir']}/weights.npz", allow_pickle=False) as z:
            return _Model({k: z[k] for k in z.files}, meta["device"])
