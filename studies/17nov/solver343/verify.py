"""Independent original-space equations and source-only prediction guards."""
import numpy as np
from scipy.special import logsumexp, softmax


def objective(x, y, theta):
    w, bias = theta[:, :-1], theta[:, -1]
    logits = x @ w.T + bias
    probabilities = softmax(logits, axis=1)
    loss = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])
    loss += np.square(w).sum() / (2*.3*len(y))
    residual = probabilities.copy()
    residual[np.arange(len(y)), y] -= 1
    residual /= len(y)
    gradient = np.column_stack((residual.T @ x + w / (.3*len(y)), residual.sum(0)))
    return float(loss), float(abs(gradient).max()), probabilities


def parity(candidate, reference):
    gap = np.sort(reference, axis=1)[:, -1] - np.sort(reference, axis=1)[:, -2]
    changes = candidate.argmax(1) != reference.argmax(1)
    return dict(max_probability_error=float(abs(candidate-reference).max()),
                label_changes=int(changes.sum()), non_tied_changes=int((changes & (gap > .002)).sum()))
