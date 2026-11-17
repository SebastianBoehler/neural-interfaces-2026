"""Terminal tie-breaking restricted to the two cognitive commands."""
import numpy as np

from inputs import require


def odds(probability):
    p = np.asarray(probability)
    require(p.ndim == 2 and p.shape[1] == 2 and np.isfinite(p).all()
            and (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1, atol=1e-8, rtol=0),
            'invalid two-command probabilities')
    logp = np.log(np.clip(p, 1e-6, 1))
    return np.clip(logp[:, 1] - logp[:, 0], -4, 4)


def decisions(scores, probability, in_domain, external, source_weight, external_weight):
    winners = scores.argmax(1)
    if source_weight == external_weight == 0:
        return winners
    margin = scores[:, 2] - scores[:, 1]
    p = np.clip(scores[probability, 1:], 1e-6, 1)
    margin[probability] = np.log(p[:, 1]) - np.log(p[:, 0])
    corrected = margin + source_weight * in_domain + external_weight * external
    return np.where(winners == 0, 0, 1 + (corrected > 0))
