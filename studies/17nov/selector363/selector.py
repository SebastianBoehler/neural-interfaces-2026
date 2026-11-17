"""A fixed causal pairwise selector and its source-rate-matched margin control."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def four_margins(base, extra, a, b):
    whole_a = np.sort(base, axis=1)[:, -1] - np.sort(base, axis=1)[:, -2]
    whole_b = np.sort(extra, axis=1)[:, -1] - np.sort(extra, axis=1)[:, -2]
    rows = np.arange(len(base))
    return np.column_stack((whole_a, whole_b, base[rows, a] - base[rows, b],
                            extra[rows, b] - extra[rows, a]))


def fit(features, labels, weights):
    scale = StandardScaler().fit(features)
    model = LogisticRegression(C=.1, max_iter=2000, tol=1e-8)
    model.fit(scale.transform(features), labels, sample_weight=weights / weights.mean())
    if (model.classes_.tolist() != [False, True] or model.n_iter_.max() >= 2000
            or not np.isfinite(model.coef_).all()):
        raise ValueError('binary source fit changed')
    return scale, model


def predict(model, x):
    scale, lr = model
    return lr.decision_function(scale.transform(x)) > 0


def margin_threshold(margin, row_weights, oof_decisions):
    """Source-only expected rate. The held rate is allowed to differ."""
    target = np.dot(row_weights, oof_decisions)
    ranking = np.argsort(-margin, kind='stable')
    ordered = margin[ranking]
    change = np.r_[True, ordered[1:] != ordered[:-1]]
    thresholds = np.r_[np.inf, ordered[change]]
    cumulative = np.r_[0, np.cumsum(row_weights[ranking])]
    stops = np.r_[np.flatnonzero(change)[1:], len(ordered)]
    mass = np.r_[0, cumulative[stops]]
    return float(thresholds[np.argmin(np.abs(mass-target))]), float(target/row_weights.sum())
