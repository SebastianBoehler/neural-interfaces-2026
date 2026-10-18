"""Candidate decoders on precomputed per-window features.

Every model exposes ``fit(train_idx)`` and ``predict_stream(session_idx)``.
``predict_stream`` receives one session's windows in acquisition order and
must use only window t and earlier windows for prediction t, without labels.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (band_covariances, causal_references, expm_sym, logm,
                      slow_waveform, whiten_tangent)


def make_clf(kind, C):
    if kind == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


class MeanLinear:
    """Torch-Linear proxy: temporal channel means -> logistic regression."""

    def __init__(self, d, C=1.0):
        self.d, self.F = d, d["X"].mean(-1)
        self.clf = make_clf("lr", C)

    def fit(self, idx):
        self.clf.fit(self.F[idx], self.d["y"][idx])

    def predict_stream(self, idx):
        return self.clf.predict(self.F[idx])


class TangentSpace:
    """Filter-bank covariances -> optional session recentering -> tangent LR.

    align:
      "none"    global training reference for every window.
      "session" training sessions recentered by their own mean; at test the
                reference is the causal running mean of the session so far,
                blended with the global training reference (``k`` windows).
    train_causal: recenter training sessions with the same causal scheme
                used at test time instead of the full-session mean.
    slow: "mean" (channel means), "wave"/"wave1" (2 Hz / 1 Hz low-pass
          waveform) or "wave1d" (1 Hz waveform minus its window mean, i.e.
          without the per-window DC offset).
    k_slow: prior weight for centring the slow features (default ``k``).
    """

    _logs = {}

    def __init__(self, d, bands=("alpha", "beta1", "beta2"), align="session",
                 k=5.0, train_causal=False, clf="lr", C=0.1, means=None,
                 stretch=False, slow="mean", k_slow=None):
        self.d, self.bands, self.align, self.means = d, bands, align, means
        self.M = (d["X"].mean(-1) if slow == "mean" else
                  slow_waveform(d["X"], cutoff=2.0 if slow == "wave" else 1.0))
        if slow == "wave1d":
            W = self.M.reshape(len(self.M), d["X"].shape[1], -1)
            self.M = (W - W.mean(-1, keepdims=True)).reshape(len(W), -1)
        self.k, self.train_causal, self.stretch = k, train_causal, stretch
        self.k_slow = k if k_slow is None else k_slow
        self.clf = make_clf(clf, C)
        self.covs = [band_covariances(d["X"], b) for b in bands]
        self.logs = [self._log(b, c) for b, c in zip(bands, self.covs)]

    @classmethod
    def _log(cls, band, C):
        if band not in cls._logs:
            cls._logs[band] = logm(C)
        return cls._logs[band]

    def _features(self, idx, refs, causal=False):
        out = []
        for C, R in zip(self.covs, refs):
            V = whiten_tangent(C[idx], R)
            if self.stretch:  # unit mean squared distance to the reference
                d2 = (V ** 2).sum(1)
                n = np.arange(1, len(V) + 1)
                disp = ((self.k * d2.mean() + np.cumsum(d2)) / (self.k + n)
                        if causal else np.full(len(V), d2.mean()))
                V = V / np.sqrt(disp)[:, None]
            out.append(V)
        return np.concatenate(out, axis=1)

    def _session_feats(self, idx, causal):
        F = self._tangent(idx, causal)
        if self.means is None:
            return F
        M = self.M[idx]
        if self.means == "center":  # session mean so far, or whole session
            n = np.arange(1, len(idx) + 1)[:, None]
            M = M - (np.cumsum(M, 0) / (self.k_slow + n) if causal else M.mean(0))
        return np.concatenate([F, M], axis=1)

    def _tangent(self, idx, causal):
        if self.align == "none":
            return self._features(idx, self.glob)
        if causal:
            refs = [causal_references(L[idx], g, self.k)
                    for L, g in zip(self.logs, self.log_glob)]
        else:
            refs = [expm_sym(L[idx].mean(0)) for L in self.logs]
        return self._features(idx, refs, causal)

    def fit(self, idx):
        self.log_glob = [L[idx].mean(0) for L in self.logs]
        self.glob = [expm_sym(g) for g in self.log_glob]
        streams = self.d["stream"][idx]
        feats, ys = [], []
        for s in dict.fromkeys(streams):
            sub = idx[streams == s]
            feats.append(self._session_feats(sub, self.train_causal))
            ys.append(self.d["y"][sub])
        self.clf.fit(np.concatenate(feats), np.concatenate(ys))

    def predict_proba_stream(self, idx):
        return self.clf.predict_proba(self._session_feats(idx, causal=True))

    def predict_stream(self, idx):
        return self.clf.predict(self._session_feats(idx, causal=True))
