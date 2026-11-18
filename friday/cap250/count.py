"""Read the stock urn's predictive class prior separately from current likelihood."""
import numpy as np
from scipy.special import logsumexp


def normalize(x):
    return x / x.sum(axis=-1, keepdims=True)


def predictive_prior(urn, p, lp, reset, cap, exponent):
    clipped = np.clip(p, 1e-9, 1.)
    # The authenticated urn has a fixed exponent of 2. Its input transform
    # changes the likelihood for the history arm without a second urn implementation.
    pred = urn(p if exponent == 2 else np.sqrt(clipped), reset, cap)
    past = np.vstack((np.zeros(len(lp)), np.cumsum(np.log(pred.sum(2)), axis=0)[:-1]))
    evidence = past + lp
    weights = np.exp(evidence - logsumexp(evidence, axis=1, keepdims=True))
    weighted = (weights[..., None] * pred).sum(1)
    return normalize(weighted / clipped**exponent), normalize(weighted)


def emissions(urn, p, lp, reset, cap):
    stock_prior, stock = predictive_prior(urn, p, lp, reset, cap, 2)
    soft_prior, _ = predictive_prior(urn, p, lp, reset, cap, 1)
    likelihood = np.clip(p, 1e-9, 1.)
    np.testing.assert_allclose(stock_prior[0], 1/3, atol=1e-12, rtol=0)
    np.testing.assert_allclose(soft_prior[0], 1/3, atol=1e-12, rtol=0)
    return stock, {
        "primary": normalize(likelihood * stock_prior),
        "history": normalize(likelihood * soft_prior),
        "confidence": normalize(np.sqrt(stock)),
    }


def terminal_scores(terminal, filtered, emitted, personal, weight):
    """Replace the live count input to CP while preserving its original LDA evidence."""
    assert np.array_equal(terminal[~personal], filtered[~personal])
    old_log = np.log(np.clip(filtered[personal], 1e-6, 1.))
    personal_log = terminal[personal] - old_log
    q = np.exp(personal_log / weight[personal, None])
    assert np.isfinite(q).all() and np.all(q <= 1+1e-9) and np.all(q >= 1e-6-1e-10)
    assert np.all(np.abs(q.sum(1)-1) <= 3.01e-6)
    np.testing.assert_allclose(old_log + personal_log, terminal[personal], atol=1e-12, rtol=0)
    scores = emitted.copy()
    scores[personal] = np.log(np.clip(emitted[personal], 1e-6, 1.)) + personal_log
    untouched = np.all(emitted == filtered, axis=1)
    scores[untouched] = terminal[untouched]
    return scores
