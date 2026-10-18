"""Probability-averaging ensemble of harness models (same causal contract).

Each member is fitted on the same training windows and streams the same
test session; the ensemble returns the weighted mean of the members' class
probabilities, so it stays causal and label-free when every member is.
"""

import numpy as np


class Ensemble:
    def __init__(self, members, weights=None):
        self.members = list(members)
        w = np.ones(len(self.members)) if weights is None else np.asarray(weights, float)
        self.w = w / w.sum()

    def fit(self, idx):
        for m in self.members:
            m.fit(idx)

    def predict_proba_stream(self, idx):
        return sum(w * m.predict_proba_stream(idx)
                   for w, m in zip(self.w, self.members))

    def predict_stream(self, idx):
        return self.predict_proba_stream(idx).argmax(1)
