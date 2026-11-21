"""Session-only personal 256-PC logistic and slow-waveform LDA heads."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression

CHECKPOINTS = {40: .25, 70: .40, 100: .50, 140: .60}
N_VIEW = (256, 41 * 32)
LOG_FLOOR = np.log(1e-6)


def selected_rows(labels, confidence):
    quotas = [np.flatnonzero(labels == k) for k in range(3)]
    count = min(int(.7 * len(rows)) for rows in quotas)
    return np.sort(np.concatenate([rows[np.argsort(-confidence[rows], kind="stable")[:count]]
                                   for rows in quotas]))


def slow_lda(x, y, weight):
    """Weighted equal-prior LDA with 0.7 covariance shrinkage, solved in sample space."""
    groups = np.eye(3)[y] * weight[:, None]
    groups /= groups.sum(0)
    means = groups.T @ x
    residual = (x - means[y]) * np.sqrt(groups.sum(1, keepdims=True) / 3)
    shrink = .7
    scale = max(shrink * np.square(residual).sum() / x.shape[1], 1e-9)
    basis = residual * np.sqrt((1 - shrink) / scale)
    lhs = basis @ basis.T + np.eye(len(basis))
    dual = cho_solve(cho_factor(lhs, check_finite=False), basis @ means.T, check_finite=False)
    coef = (means - (basis.T @ dual).T) / scale
    return coef, -.5 * np.sum(coef * means, axis=1)


def fit_heads(pc, slow, y, weight):
    logit = LogisticRegression(C=.1, max_iter=200).fit(pc, y, sample_weight=weight)
    if not np.array_equal(logit.classes_, np.arange(3)) or logit.n_iter_.max() >= 200:
        raise ValueError("self-enrolment logistic head did not fit all three classes")
    return ((logit.coef_, logit.intercept_), slow_lda(slow, y, weight))


def log_evidence(heads, views):
    return sum(np.maximum(log_softmax(x @ coef.T + bias, axis=-1), LOG_FLOOR)
               for (coef, bias), x in zip(heads, views, strict=True)) / 2


def personal_fit(pc, slow, valid, labels, confidence):
    """Fit on the arrived prefix. Omit each selected label once for the agreement guard."""
    views = tuple(np.asarray(view, dtype=np.float64) for view in (pc, slow))
    size = len(views[0])
    valid = np.asarray(valid, dtype=bool)
    if (tuple(view.shape for view in views) != tuple((size, d) for d in N_VIEW)
            or labels.shape != (size,) or confidence.shape != (size,) or valid.shape != (size,)):
        raise ValueError("self-enrolment prefix shapes differ")
    if (not np.isin(labels, (0, 1, 2)).all() or not np.isfinite(confidence).all()
            or (confidence <= 0).any() or (confidence > 1).any()):
        raise ValueError("invalid self-enrolment labels or confidence")
    rows = np.flatnonzero(valid)
    chosen = rows[selected_rows(labels[rows], confidence[rows])]
    if len(chosen) < 6:
        return None, np.nan, len(chosen)
    norm = [(x[valid].mean(0), np.maximum(x[valid].std(0), floor))
            for x, floor in zip(views, (1e-3, np.sqrt(1e-9)), strict=True)]
    a, b = [(view[chosen] - mu) / sd for view, (mu, sd) in zip(views, norm, strict=True)]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("nonfinite self-enrolment features")
    y, w = labels[chosen], confidence[chosen]
    hits = []
    all_rows = np.arange(len(chosen))
    for i in all_rows:
        keep = all_rows != i
        heads = fit_heads(a[keep], b[keep], y[keep], w[keep])
        hits.append(log_evidence(heads, (a[i], b[i])).argmax() == y[i])
    agree = float(np.mean(hits))
    model = (norm, fit_heads(a, b, y, w)) if agree >= .75 else None
    return model, agree, len(chosen)


def personal_log(model, pc, slow):
    norm, heads = model
    return log_evidence(heads, [(x - mu) / sd for x, (mu, sd) in zip((pc, slow), norm, strict=True)])
