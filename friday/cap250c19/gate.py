"""Fixed odds, previous-window sham and equal log blend on one causal session."""

import numpy as np

ODDS = 1.5


def splice(stock, challenger, accept):
    out = stock.copy()
    out[accept] = challenger[accept]
    np.testing.assert_array_equal(out[~accept], stock[~accept])
    return out


def decide(stock, challenger):
    a, b = stock["scores"], challenger["scores"]
    row = np.arange(len(a))
    baseline, competing = a.argmax(1), b.argmax(1)
    prior = np.maximum(0, row - 1)
    eligible = (row >= 91) & (baseline != competing)
    # A unique maximum is required for both systems before invoking their odds.
    eligible &= (np.sum(a == a.max(1, keepdims=True), axis=1) == 1)
    eligible &= (np.sum(b == b.max(1, keepdims=True), axis=1) == 1)
    for tape in (stock, challenger):
        q = tape["q"]
        eligible &= tape["valid"] & tape["valid"][prior]
        for ix in (row, prior):
            eligible &= (q[ix, baseline] > 0) & (q[ix, competing] > 0)

    live, sham = eligible.copy(), eligible.copy()
    for tape in (stock, challenger):
        q = tape["q"]
        live &= q[row, competing] >= ODDS * q[row, baseline]
        sham &= q[prior, competing] >= ODDS * q[prior, baseline]
    np.testing.assert_array_equal(splice(a, b, np.zeros(len(a), dtype=bool)), a)
    np.testing.assert_array_equal(splice(a, b, np.ones(len(a), dtype=bool)), b)
    return dict(eligible=eligible, live=live, sham_mask=sham, gate=splice(a, b, live),
                sham=splice(a, b, sham), blend=splice(a, .5 * (a + b), eligible))
