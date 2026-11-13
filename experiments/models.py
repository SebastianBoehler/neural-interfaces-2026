"""Candidate decoders on precomputed per-window features.

Every model exposes ``fit(train_idx)`` and ``predict_stream(session_idx)``.
``predict_stream`` receives one session's windows in acquisition order and
must use only window t and earlier windows for prediction t, without labels.
"""

import numpy as np
from scipy.linalg import expm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (BANDS, band_covariances, causal_references, expm_sym, logm,
                      slow_waveform, whiten_tangent)


def make_clf(kind, C):
    if kind == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def rotate_features(F, n, n_bands, eps, copies, rng):
    """F plus ``copies`` copies with every window's channels rotated by its own
    random orthogonal Q = exp(eps (A - A^T)), as dg.rotate does to signals: the
    tangent matrices of every band become Q S Q^T, the slow waveform Q M."""
    iu = np.triu_indices(n)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    size = len(coef)
    out = [F]
    for _ in range(copies):
        A = eps * rng.standard_normal((len(F), n, n))
        Q = np.stack([expm(a - a.T) for a in A])
        G = F.copy()
        for b in range(n_bands):
            S = np.zeros((len(F), n, n))
            S[:, iu[0], iu[1]] = F[:, b * size:(b + 1) * size] / coef
            S = S + np.triu(S, 1).transpose(0, 2, 1)
            S = Q @ S @ Q.transpose(0, 2, 1)
            G[:, b * size:(b + 1) * size] = S[:, iu[0], iu[1]] * coef
        if F.shape[1] > n_bands * size:  # slow features: channel-major waveform
            M = F[:, n_bands * size:].reshape(len(F), n, -1)
            G[:, n_bands * size:] = (Q @ M).reshape(len(F), -1)
        out.append(G)
    return np.concatenate(out)


class MeanLinear:
    """Torch-Linear proxy: temporal channel means -> logistic regression."""

    def __init__(self, d, C=1.0):
        self.d, self.F = d, d["X"].mean(-1)
        self.clf = make_clf("lr", C)

    def fit(self, idx):
        from protocol import require_honoured; require_honoured(self.d, idx, set(), "MeanLinear")
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
                 stretch=False, slow="mean", k_slow=None, wscale=False, group=None,
                 dro_eta=1.0, dro_iter=5, rot=0.0, rot_copies=2):
        self.d, self.bands, self.align, self.means = d, bands, align, means
        self.M = (d["X"].mean(-1) if slow == "mean" else
                  slow_waveform(d["X"], cutoff=2.0 if slow == "wave" else 1.0))
        if slow == "wave1d":
            W = self.M.reshape(len(self.M), d["X"].shape[1], -1)
            self.M = (W - W.mean(-1, keepdims=True)).reshape(len(W), -1)
        self.k, self.train_causal, self.stretch = k, train_causal, stretch
        self.wscale = wscale
        self.group, self.dro_eta, self.dro_iter = group, dro_eta, dro_iter  # R8
        self.rot, self.rot_copies = rot, rot_copies  # rotated training copies (R15)
        self.k_slow = k if k_slow is None else k_slow
        self.clf = make_clf(clf, C)
        tag = d.get("tag", "")
        self.covs = [band_covariances(d["X"], b, tag) for b in bands]
        self.logs = [self._log((tag, b), c) for b, c in zip(bands, self.covs)]

    @classmethod
    def _log(cls, key, C):
        if key not in cls._logs:
            cls._logs[key] = logm(C)
        return cls._logs[key]

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

    def _low_columns(self):
        """Feature columns carrying information below 4 Hz (delta, slow)."""
        n = len(self.covs[0][0])
        size = n * (n + 1) // 2
        cols = [np.arange(i * size, (i + 1) * size)
                for i, b in enumerate(self.bands) if BANDS[b][0] < 4]
        if self.means is not None:
            start = size * len(self.bands)
            cols.append(np.arange(start, start + self.M.shape[1]))
        return np.concatenate(cols) if cols else np.zeros(0, int)

    def fit(self, idx):
        from protocol import require_honoured; require_honoured(self.d, idx, {"w", "m"}, "TangentSpace")
        nolow = self.d.get("nolow", np.zeros(len(self.d["y"]), bool))
        hi = idx[~nolow[idx]]  # global low-band references from full-band data
        ref = idx
        if self.wscale:  # target statistics: global references from PROTEUS
            ref = hi = idx[self.d["split"][idx] != "external"]
        self.log_glob = [L[hi if BANDS[b][0] < 4 else ref].mean(0)
                         for b, L in zip(self.bands, self.logs)]
        self.glob = [expm_sym(g) for g in self.log_glob]
        low = self._low_columns()
        streams = self.d["stream"][idx]
        feats, ys, ws = [], [], []
        for s in dict.fromkeys(streams):
            sub = idx[streams == s]
            feats.append(self._session_feats(sub, self.train_causal))
            if nolow[sub].all():  # missing below 4 Hz: no gradient on those
                feats[-1][:, low] = 0.0
            ys.append(self.d["y"][sub])
            ws.append(self.d["w"][sub])
        y = np.concatenate(ys)
        kw = {}
        if self.d.get("tag"):  # external data: class-balanced, weighted samples
            w = np.concatenate(ws)
            cw = w.sum() / (3 * np.bincount(y, weights=w, minlength=3))
            kw["logisticregression__sample_weight"] = w * cw[y]
            if self.wscale:  # target statistics: scaler weighted like the loss
                kw["standardscaler__sample_weight"] = w
        F = np.concatenate(feats)
        if self.rot:
            assert not kw, "rotation copies are for PROTEUS-only training"
            F = rotate_features(F, len(self.covs[0][0]), len(self.bands), self.rot,
                                self.rot_copies, np.random.default_rng(0))
            y = np.tile(y, self.rot_copies + 1)
        if self.group:  # participant-balanced weights, or GroupDRO reweighting
            assert not kw, "participant weighting is for PROTEUS-only training"
            g = np.unique(np.concatenate([self.d["subject"][idx[streams == s]]
                                          for s in dict.fromkeys(streams)]),
                          return_inverse=True)[1]
            q = np.ones(g.max() + 1) / (g.max() + 1)
            for _ in range(self.dro_iter if self.group == "dro" else 1):
                w = q[g] / np.bincount(g)[g]  # each participant's mass is q_g
                self.clf.fit(F, y, logisticregression__sample_weight=w * len(y))
                if self.group == "dro":
                    p = self.clf.predict_proba(F)[np.arange(len(y)), y]
                    loss = np.bincount(g, weights=-np.log(np.clip(p, 1e-9, 1))) / np.bincount(g)
                    q = q * np.exp(self.dro_eta * loss)
                    q /= q.sum()
            return
        self.clf.fit(F, y, **kw)

    def predict_proba_stream(self, idx):
        return self.clf.predict_proba(self._session_feats(idx, causal=True))

    def predict_stream(self, idx):
        return self.clf.predict(self._session_feats(idx, causal=True))


class ReveLinear:
    """Frozen REVE embeddings (scripts/extract_reve.py) -> logistic regression.

    center: subtract the session mean embedding (whole session for training,
    causal running mean with ``k`` zero pseudo-windows at test time).
    """

    def __init__(self, d, center=True, k=5.0, C=0.1):
        from features import CACHE
        self.d, self.center, self.k = d, center, k
        self.E = np.load(CACHE / "reve_base.npy").astype(np.float64)
        self.clf = make_clf("lr", C)

    def _feats(self, idx, causal):
        E = self.E[idx]
        if not self.center:
            return E
        if causal:
            n = np.arange(1, len(idx) + 1)[:, None]
            return E - np.cumsum(E, 0) / (self.k + n)
        return E - E.mean(0)

    def fit(self, idx):
        from protocol import require_honoured; require_honoured(self.d, idx, set(), "ReveLinear")
        st = self.d["stream"][idx]
        F = [self._feats(idx[st == s], False) for s in dict.fromkeys(st)]
        y = [self.d["y"][idx[st == s]] for s in dict.fromkeys(st)]
        self.clf.fit(np.concatenate(F), np.concatenate(y))

    def predict_proba_stream(self, idx):
        return self.clf.predict_proba(self._feats(idx, True))

    def predict_stream(self, idx):
        return self.predict_proba_stream(idx).argmax(1)
