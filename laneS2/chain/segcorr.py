"""Lane K test 2: bounded Gaussian-density correction of post-RouteB raw posteriors from segment-dispersion features.
Pure numpy, causal per person in delivery order. Registered form in evidence/wave_20nov/laneK_council_tests.md."""
import numpy as np

W, N0, MIN_EFF, CLIP, PRIOR = 0.10, 20.0, 8, 5.0, 20.0


def features(seg, kind):
    """seg: n x 4 x 3 segment logit contributions of one member. whole = centred whole-window logits (n x 3); disp = + class-wise SD over segments."""
    z = seg.sum(1)
    z = z - z.mean(1, keepdims=True)
    if kind == "whole":
        return z
    d = seg.std(1)
    return np.concatenate([z, d], 1)


def correct(raw, labels, x_own, x_hist=None, w=W):
    """raw: n x 3 posteriors in delivery order; labels: n frozen shadow pseudo-labels; x_own: n x f features of this stream.
    x_hist: features the densities are fitted on (default x_own = genuine history; donor = another person's stream, same age).
    Density at window t uses windows < t (labels and x_hist). Returns corrected posteriors and the boolean 'active' mask."""
    x_hist = x_own if x_hist is None else x_hist
    n, f = x_own.shape
    out, active = raw.copy(), np.zeros(n, bool)
    cnt, sm, sq = np.zeros(3), np.zeros((3, f)), np.zeros((3, f))
    for t in range(n):
        if (cnt >= MIN_EFF).all():
            tot = cnt.sum()
            mu0 = sm.sum(0) / tot
            mu = (sm + PRIOR * mu0) / (cnt[:, None] + PRIOR)
            within = (sq - 2 * mu * sm + cnt[:, None] * mu ** 2).sum(0)
            var0 = sq.sum(0) / tot - mu0 ** 2
            v = np.maximum((within + N0 * var0) / (tot + N0), 1e-6)
            ll = -0.5 * (((x_own[t] - mu) ** 2) / v).sum(1)
            ll = np.clip(ll - ll.mean(), -CLIP, CLIP)
            p = raw[t] * np.exp(w * ll)
            out[t] = p / p.sum()
            active[t] = True
        k = labels[t]
        cnt[k] += 1; sm[k] += x_hist[t]; sq[k] += x_hist[t] ** 2
    return out, active
