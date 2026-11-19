"""NI-V statistics: paired delta with people-nested bootstrap (people AND orders resampled), decisions changed,
null calibration and the sealed-weighted combination.  Pure numpy; works on bank.py outputs.

delta_summary(A, B): A, B are cell arrays (scheme, order, person, visit) in %. Person value = mean over visits;
a bootstrap draw resamples people with replacement and, independently inside each scheme, the orders with replacement;
the statistic is the mean over drawn orders and drawn people (person-equal) of the chosen schemes.
Headline numbers use the full-session shuffle only (PRIMARY); within-run is reported separately (SECONDARY).
"""
import numpy as np

DRAWS = 10000


def person_table(cell):
    """(scheme, order, person) = mean over visits and (only) finite cells."""
    return np.nanmean(cell, axis=3)


def boot(delta, rng, draws=DRAWS):
    """delta (scheme, order, person) -> draws of the pooled statistic."""
    ns, no, npers = delta.shape
    pi = rng.integers(npers, size=(draws, npers))
    out = np.zeros(draws)
    for s in range(ns):
        oi = rng.integers(no, size=(draws, no))
        out += delta[s][oi[:, :, None], pi[:, None, :]].mean((1, 2))
    return out / ns


def interval(x, level):
    lo = (1 - level) / 2 * 100
    return [float(np.percentile(x, lo)), float(np.percentile(x, 100 - lo))]


PRIMARY, SECONDARY = (0,), (1,)  # scheme index: 0 full-session shuffle (PRIMARY), 1 within-run shuffle (secondary)


def delta_summary(cell_a, cell_b, seed=19, draws=DRAWS, schemes=PRIMARY):
    d = (person_table(cell_a) - person_table(cell_b))[list(schemes)]
    rng = np.random.default_rng(seed)
    b = boot(d, rng, draws)
    per_scheme = d.mean((1, 2))
    return dict(pp=float(d.mean()), se=float(b.std()), i80=interval(b, .8), i95=interval(b, .95),
                p_pos=float((b > 0).mean()), by_scheme=per_scheme.tolist(),
                by_order=d.mean(2).tolist(), by_person=d.mean((0, 1)).tolist(),
                n_people=d.shape[2], mdd=float(2.49 * b.std()))


def changed(pred_a, pred_b, y):
    """Decisions changed between two prediction banks (scheme, order, rows); both defined only."""
    ok = (pred_a >= 0) & (pred_b >= 0)
    diff = ok & (pred_a != pred_b)
    gain = diff & (pred_a == y) & (pred_b != y)
    loss = diff & (pred_b == y) & (pred_a != y)
    n = ok.sum()
    return dict(windows=int(n), changed=int(diff.sum()), frac=float(diff.sum() / max(n, 1)),
                a_right_b_wrong=int(gain.sum()), b_right_a_wrong=int(loss.sum()))


def report(cell_a, cell_b, draws=DRAWS):
    """Headline = full-session shuffle only (fresh orders, none run-contiguous); within-run is secondary."""
    return dict(primary_full=delta_summary(cell_a, cell_b, draws=draws, schemes=PRIMARY),
                secondary_within=delta_summary(cell_a, cell_b, draws=draws, schemes=SECONDARY))


def null_summary(cell_a, cell_a2, seed=23, draws=DRAWS, schemes=PRIMARY):
    """Null: identical configuration, different seed. Reports the pooled delta, bootstrap SE and the minimal
    detectable delta (one-sided alpha .05, power .80: 2.49 SE), and its largest per-order absolute mean."""
    s = delta_summary(cell_a, cell_a2, seed, draws, schemes)
    d = (person_table(cell_a) - person_table(cell_a2))[list(schemes)]
    s["max_abs_order"] = float(np.abs(d.mean(2)).max())
    s["person_sd"] = float(d.mean((0, 1)).std(ddof=1))
    return s


def combine(new, ret, w_new=0.57):
    """Sealed-weighted pooled delta from the cohort deltas (pp)."""
    return w_new * new + (1 - w_new) * ret


def combine_boot(draws_new, draws_ret, w_new=0.57):
    x = w_new * draws_new + (1 - w_new) * draws_ret
    return dict(pp=float(x.mean()), i80=interval(x, .8), i95=interval(x, .95), p_pos=float((x > 0).mean()))
