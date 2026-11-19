"""Defer a personalized covariance until a refit or a live readout uses it."""
import numpy as np


class PendingCovariance:
    __slots__ = ('window', 'value', 'row')

    def __init__(self, row, window):
        self.window = window.copy()
        self.value = None
        self.row = row

    def get(self, compute, resolved):
        if self.window is not None:
            self.value = compute(self.window)
            self.window = None
            resolved.append(self.row)
        return self.value


def instrument(stage, lazy):
    original_feats = stage.feats
    original_select = stage.select
    original_refit = stage.refit
    original_predict = stage.predict
    stats = dict(cov=[], selected=[], consumes=[], queries=[])

    def select(labels, confidence):
        rows = original_select(labels, confidence)
        stats['selected'].append(rows.tolist())
        if len(np.unique(labels[rows])) >= 2:
            stats['consumes'] += rows.tolist()
        return rows

    def feats(window):
        row = stage.t - 1
        if lazy:
            return PendingCovariance(row, window)
        stats['cov'].append(row)
        return original_feats(window)

    def refit(stop):
        p = np.asarray(stage.P[:stop])
        labels, confidence = p.argmax(1), p.max(1)
        chosen = stage.select(labels, confidence)
        if len(np.unique(labels[chosen])) < 2:
            stage.nfail += 1
            return None
        # The original eigensolver and LDA only access the chosen rows.
        history, posterior = stage.C, stage.P
        stage.C = [history[i].get(original_feats, stats['cov']) for i in chosen]
        stage.P = [posterior[i] for i in chosen]
        stage.select = lambda *_: np.arange(len(chosen))
        try:
            return original_refit(len(chosen))
        finally:
            stage.C, stage.P, stage.select = history, posterior, select

    def predict(value):
        if stage.model is not None:
            stats['queries'].append(stage.t - 1)
            if lazy:
                value = value.get(original_feats, stats['cov'])
        return original_predict(value)

    stage.feats = feats
    stage.select = select
    if lazy:
        stage.refit = refit
    stage.predict = predict
    return stats


def detach(stage):
    """Release bound instrumentation closures at the end of a paired run."""
    for name in ('feats', 'select', 'refit', 'predict'):
        if name in vars(stage):
            delattr(stage, name)
