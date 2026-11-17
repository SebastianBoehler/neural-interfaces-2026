"""Matched power/full-covariance features with source-only and causal references."""
import hashlib
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BANDS = ((4, 8), (6, 10), (8, 12), (10, 14), (12, 16), (14, 18),
         (16, 20), (18, 24), (20, 28), (24, 32), (28, 36), (32, 40))


class Geometry:
    def __init__(self, x, root):
        sys.path.insert(0, str(root / 'experiments'))
        import features as ft
        self.ft = ft
        assert x.shape[1:] == (19, 480) and np.isfinite(x).all()
        x = ft.bandpass(x - x.mean(1, keepdims=True), 4, 40)
        self.c, self.l = [], []
        for lo, hi in BANDS:
            b = ft.bandpass(x, lo, hi)
            for segment in (b, b[..., :240], b[..., 240:]):
                c = ft.covariances(segment, shrink=.1)
                if not np.isfinite(c).all() or np.linalg.eigvalsh(c).min() <= 0:
                    raise ValueError('Invalid covariance')
                self.c.append(c)
                self.l.append(ft.logm(c))

    def features(self, ids, reference):
        power, cov = [], []
        for c, ref in zip(self.c, reference):
            p = self.ft.invsqrtm(ref)
            if p.ndim == 2:
                logc = self.ft.logm(p @ c[ids] @ p)
            else:
                logc = self.ft.logm(np.einsum('nij,njk,nlk->nil', p, c[ids], p))
            power.append(np.diagonal(logc, axis1=1, axis2=2))
            cov.append(self.ft.upper(logc))
        result = {'power': np.concatenate(power, 1), 'cov': np.concatenate(cov, 1)}
        if not all(np.isfinite(x).all() for x in result.values()):
            raise ValueError('Nonfinite feature')
        return result

    def prior(self, ids):
        return [l[ids].mean(0) for l in self.l]

    def causal(self, ids, prior):
        n = np.arange(1, len(ids) + 1)[:, None, None]
        refs = [self.ft.expm_sym((20 * p + np.cumsum(l[ids], 0)) / (20 + n))
                for p, l in zip(prior, self.l)]
        return self.features(ids, refs)

    def training(self, ids, stream):
        out = {k: None for k in ('power', 'cov')}
        for session in dict.fromkeys(stream[ids]):
            pos = np.flatnonzero(stream[ids] == session)
            rows = ids[pos]
            refs = [self.ft.expm_sym(l[rows].mean(0)) for l in self.l]
            result = self.features(rows, refs)
            for arm in out:
                if out[arm] is None:
                    out[arm] = np.empty((len(ids), result[arm].shape[1]))
                out[arm][pos] = result[arm]
        return out


def fit(features, y):
    keep = np.isin(y, (1, 2))
    models = {}
    for arm, x in features.items():
        assert x.shape[1] == {'power': 684, 'cov': 6840}[arm]
        scaler = StandardScaler().fit(x[keep])
        head = LogisticRegression(C=.05, max_iter=2000, class_weight='balanced', tol=1e-4)
        head.fit(scaler.transform(x[keep]), y[keep] - 1)
        assert head.classes_.tolist() == [0, 1] and head.n_iter_.max() < 2000
        models[arm] = (scaler, head)
    return models


def predict(models, features):
    # Match the one-window inference interface exactly, including BLAS rounding.
    out = {k: np.concatenate([head.predict_proba(scaler.transform(row[None]))
                              for row in features[k]])
           for k, (scaler, head) in models.items()}
    assert all(np.isfinite(p).all() and np.allclose(p.sum(1), 1) for p in out.values())
    return out


def digest(models):
    return hashlib.sha256(pickle.dumps(models)).hexdigest()
