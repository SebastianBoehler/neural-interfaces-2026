"""Keep class selection and energy features while changing CSP row weights."""
from functools import partial

import numpy as np
from scipy.linalg import eigh


def shape_only(covariances):
    mean_power = np.diagonal(covariances, axis1=-2, axis2=-1).mean(-1)
    if not np.all(np.isfinite(mean_power) & (mean_power > 0)):
        raise FloatingPointError('CSP covariance has invalid mean power')
    return covariances / mean_power[..., None, None]


def refit_shape(stage, stop):
    # Keep the original refit's selector, shrinkage and eigenvector choices.
    p = np.asarray(stage.P[:stop])
    labels, confidence = p.argmax(1), p.max(1)
    chosen = stage.select(labels, confidence)
    classes = np.unique(labels[chosen])
    if len(classes) < 2:
        stage.nfail += 1
        return None
    raw = np.stack(stage.C[:stop])
    unit = shape_only(raw[chosen])
    weights, features = [], []
    for band in range(len(stage.sos)):
        rows = unit[:, band]
        directions = []
        for label in classes:
            a = labels[chosen] == label
            ma = np.average(rows[a], axis=0, weights=np.ones(a.sum()))
            mo = np.average(rows[~a], axis=0, weights=np.ones((~a).sum()))
            mo = .9*mo + .1*np.trace(mo)/len(mo)*np.eye(len(mo))
            _, vectors = eigh(ma, ma + mo)
            directions.extend((vectors[:, :2], vectors[:, -2:]))
        spatial = np.concatenate(directions, axis=1)
        weights.append(spatial)
        features.append(stage.lv(spatial, raw[chosen, band]))
    classifier = stage.LDA(solver='lsqr', shrinkage='auto')
    return weights, classifier.fit(np.concatenate(features, axis=1), labels[chosen])


def use_shape_fit(stage):
    stage.refit = partial(refit_shape, stage)
    return stage
