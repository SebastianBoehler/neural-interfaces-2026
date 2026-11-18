"""Use only current branch scores and preceding within-session confidence."""
import numpy as np

COUNT = 20
PRIOR = np.log(2.)
FLOOR = np.log(1.25)
SWITCH = .50


def log_scores(scores, probability_rows):
    assert scores.ndim == 2 and scores.shape[1] == 3
    assert np.isfinite(scores).all()
    p = scores[probability_rows]
    assert (p >= 0).all() and np.allclose(p.sum(1), 1., atol=1e-6)
    assert (scores[~probability_rows] <= 0).all()
    out = scores.astype(np.float64).copy()
    out[probability_rows] = np.log(np.clip(p, 1e-12, 1.))
    assert np.array_equal(out.argmax(1), scores.argmax(1))
    return out


def prior_scales(logits):
    ordered = np.sort(logits, axis=1)
    gaps = ordered[:, -1]-ordered[:, -2]
    past = np.r_[0., np.cumsum(gaps[:-1])]
    return np.maximum(FLOOR, (COUNT*PRIOR+past)/(COUNT+np.arange(len(logits))))


def variants(old, stock):
    a, b = old.argmax(1), stock.argmax(1)
    row = np.arange(len(a))
    old_odds = old[row, a]-old[row, b]
    stock_odds = stock[row, b]-stock[row, a]
    assert (old_odds >= 0).all() and (stock_odds >= 0).all()
    old_scale, stock_scale = (prior_scales(x) for x in (old,stock))
    assert old_scale[0] == stock_scale[0] == PRIOR
    decision = old_odds/old_scale - stock_odds/stock_scale > SWITCH
    static = (old_odds-stock_odds)/PRIOR > SWITCH
    choices = dict(stock=b, old=a, fixed=np.where(static,a,b), causal=np.where(decision,a,b))
    assert choices['fixed'][0] == choices['causal'][0]
    agree = a == b
    assert all(np.array_equal(v[agree],b[agree]) for v in choices.values())
    return choices
