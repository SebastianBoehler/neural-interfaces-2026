"""Probability-averaging ensemble of harness models (same causal contract).

Each member is fitted on the same training windows and streams the same
test session; the ensemble returns the weighted mean of the members' class
probabilities, so it stays causal and label-free when every member is.
"""

import numpy as np

import session_em


def rebalance(P, alpha, k, W=0):
    """Causal class rebalancing of one session's probabilities (rows in order).

    Row t is divided by m ** alpha, m being the mean probability of the
    preceding rows (all of them, or the last ``W`` when W > 0) blended with
    a uniform prior weighted as ``k`` rows.
    """
    out, seen = np.empty_like(P), np.zeros(P.shape[1])
    for n, p in enumerate(P):
        if W:
            h = P[max(0, n - W):n]
            m = (k / P.shape[1] + np.asarray(h).sum(0)) / (k + len(h))
        else:
            m = (k / P.shape[1] + seen) / (k + n)
        q = p / m ** alpha
        out[n] = q / q.sum()
        seen += p
    return out


class Ensemble:
    def __init__(self, members, weights=None, alpha=0.0, k=5.0, W=0, em=None):
        self.members = list(members)
        self.em = em  # session_em.SessionEM kwargs, applied after rebalancing (R19)
        w = np.ones(len(self.members)) if weights is None else np.asarray(weights, float)
        self.w = w / w.sum()
        self.alpha, self.k, self.W = alpha, k, W

    def fit(self, idx):
        for m in self.members:
            m.fit(idx)

    def predict_proba_stream(self, idx):
        P = sum(w * m.predict_proba_stream(idx)
                for w, m in zip(self.w, self.members))
        P = rebalance(P, self.alpha, self.k, self.W) if self.alpha else P
        if self.em:  # idx is one session in acquisition order
            d = self.members[0].d
            F = session_em.gaze_features(d["X"][idx].astype(np.float64), list(d["ch_names"]))
            P = session_em.apply(F, P, np.zeros(len(idx)), **self.em)
        return P

    def predict_stream(self, idx):
        return self.predict_proba_stream(idx).argmax(1)
