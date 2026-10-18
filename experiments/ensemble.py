"""Probability-averaging ensemble of harness models (same causal contract).

Each member is fitted on the same training windows and streams the same
test session; the ensemble returns the weighted mean of the members' class
probabilities, so it stays causal and label-free when every member is.
"""

import numpy as np


def rebalance(P, alpha, k):
    """Causal class rebalancing of one session's probabilities (rows in order).

    Row t is divided by m ** alpha, m being the mean probability of rows
    0..t-1 blended with a uniform prior weighted as ``k`` rows.
    """
    out, seen = np.empty_like(P), np.zeros(P.shape[1])
    for n, p in enumerate(P):
        q = p / ((k / P.shape[1] + seen) / (k + n)) ** alpha
        out[n] = q / q.sum()
        seen += p
    return out


class Ensemble:
    def __init__(self, members, weights=None, alpha=0.0, k=5.0):
        self.members = list(members)
        w = np.ones(len(self.members)) if weights is None else np.asarray(weights, float)
        self.w = w / w.sum()
        self.alpha, self.k = alpha, k

    def fit(self, idx):
        for m in self.members:
            m.fit(idx)

    def predict_proba_stream(self, idx):
        P = sum(w * m.predict_proba_stream(idx)
                for w, m in zip(self.w, self.members))
        return rebalance(P, self.alpha, self.k) if self.alpha else P

    def predict_stream(self, idx):
        return self.predict_proba_stream(idx).argmax(1)
