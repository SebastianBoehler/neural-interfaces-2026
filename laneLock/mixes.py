"""Pre-registered mixer interventions. Each output row depends on its own inputs."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ITEMS = (("c1", 1), ("hb", 1), ("1", .5), ("2", .5), ("3", 1), ("4", 1), ("10", 5),
         ("9", 1), ("sc_fbraw", 1), ("sc_fb", 1), ("sc_sinc", 1), ("sc_fbl", 1), ("sc_rest", 1))
PORTFOLIO = ("c1", "hb", "sc_fb", "sc_sinc", "sc_fbraw")
GROUPS = (("c1", "hb"), ("9", "sc_fb", "sc_sinc", "sc_fbl"), ("sc_fbraw",),
          ("1", "2", "3", "4"), ("10",), ("sc_rest",))


def pools(members, items, keys):
    chosen = [(k, w) for k, w in items if k in keys]
    assert set(keys) == {k for k, _ in chosen}
    return sum(w * members[k] for k, w in chosen) / sum(w for _, w in chosen)


def portfolio(members, items):
    mean = pools(members, items, PORTFOLIO)
    # The replay reconstructs some c1 values by subtraction.
    assert mean.min() >= -.001, "unexpected signed c1 reconstruction"
    projected = mean.clip(0)
    return projected / projected.sum(1, keepdims=True)


def features(members, items, interaction):
    stock = pools(members, items, members.keys())
    logp = np.log(stock.clip(1e-8))
    views = np.stack([pools(members, items, group) for group in GROUPS], axis=2)
    contrast = (np.log(views.clip(1e-8)) - logp[:, :, None]).clip(-2, 2)
    if interaction:
        disagree = views.argmax(1) != stock.argmax(1)[:, None]
        contrast = np.concatenate([contrast, contrast * disagree[:, None, :]], axis=2)
    return logp, contrast


def fit(logp, x, y, cells):
    """Fit weighted multinomial residuals. Clamp only when applying the fit."""
    _, inverse, counts = np.unique(cells, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse]
    weight /= weight.sum()
    target = np.eye(3)[y]
    shape = x.shape[1:]

    def objective(flat):
        coef = flat.reshape(shape)
        logits = logp + np.einsum("ncf,cf->nc", x, coef)
        ll = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.einsum("n,nc,nc->", weight, target, ll) + .04 * (flat @ flat)
        grad = np.einsum("n,nc,ncf->cf", weight, np.exp(ll) - target, x)
        return loss, grad.ravel() + .08 * flat

    result = minimize(objective, np.zeros(np.prod(shape)), method="L-BFGS-B", jac=True)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"stack fit failed: {result.message}")
    return result.x.reshape(shape)


def mix_hook(fn):
    """Validate the live item roster, posterior sums and rowwise prefix behavior."""
    def run(members, items, perm):
        keymap = {str(k): k for k, _ in items}
        assert tuple((str(k), w) for k, w in items) == ITEMS
        active = {k: members[src] for k, src in keymap.items()}
        result = fn(active, perm)
        first = {k: p[:37] for k, p in active.items()}
        np.testing.assert_allclose(fn(first, perm[:37]), result[:37], atol=1e-14, rtol=0)
        assert np.isfinite(result).all() and (result >= 0).all()
        np.testing.assert_allclose(result.sum(1), 1, atol=1e-7, rtol=0)
        return result
    return run


def learned(coef, person, interaction):
    def run(members, perm):
        logp, x = features(members, ITEMS, interaction)
        batch_coef = coef[person[perm]]
        adjustment = np.einsum("ncf,ncf->nc", x, batch_coef).clip(-.5, .5)
        logits = logp + adjustment
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return run
