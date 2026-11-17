"""Fractional CSP covariance estimation with the original hard-label LDA."""
import numpy as np
from scipy.linalg import eigh


def fractional_type(parent):
    class FractionalCSP(parent):
        def refit(self, t):
            past = np.asarray(self.P[:t])
            y, confidence = past.argmax(1), past.max(1)
            chosen = self.select(y, confidence)
            classes = np.unique(y[chosen])
            if len(classes) < 2:
                return super().refit(t)
            membership = past[chosen][:, classes]
            membership /= membership.sum(1, keepdims=True)
            membership = .5 * membership + .5 * (y[chosen, None] == classes)
            if not np.isfinite(membership).all() or not np.allclose(membership.sum(1), 1, atol=1e-12):
                raise ValueError('invalid fractional class assignment')
            C = np.stack(self.C[:t])
            spatial, feats = [], []
            for band in range(len(self.sos)):
                band_cov, filters = C[chosen, band], []
                for w in membership.T:
                    positive = np.average(band_cov, axis=0, weights=w)
                    negative = np.average(band_cov, axis=0, weights=1-w)
                    negative = .9 * negative + .1 * np.trace(negative) / len(negative) * np.eye(len(negative))
                    _, basis = eigh(positive, positive + negative)
                    filters += [basis[:, :2], basis[:, -2:]]
                weights = np.concatenate(filters, axis=1)
                spatial.append(weights)
                feats.append(self.lv(weights, band_cov))
            readout = self.LDA(solver='lsqr', shrinkage='auto').fit(np.concatenate(feats, 1), y[chosen])
            self.fractional_calls += 1
            return spatial, readout
    return FractionalCSP


def apply(factory, config, features, raw, filtered, sessions, recognized=None):
    result = np.full_like(raw, np.nan, dtype=np.float64)
    stats = []
    for session in sessions:
        model = factory(config)
        model.fractional_calls = 0
        # Only the immutable window-local feature computation is shared in replay.
        model.feats = features.__getitem__
        for i, row in enumerate(session):
            result[row] = model.step(row, raw[row], filtered[row],
                                     recognized is not None and i >= recognized)
        if model.log['err'] or model.dead or model.t != len(session):
            raise ValueError(f'original CP error path activated: {model.log}')
        stats.append(dict(refits=model.refits, fused=model.fused, stopped=model.stopped,
                          fractional_calls=model.fractional_calls, single_class=model.nfail))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite full-chain replay')
    return result, stats
