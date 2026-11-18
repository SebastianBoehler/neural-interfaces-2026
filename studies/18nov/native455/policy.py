"""Correct the native CSP-LDA prior and measure its past agreement with the chain."""
import numpy as np


def corrections(raw, filtered, terminal, sessions):
    """Return additive log corrections for a full bank in original row order."""
    assert raw.shape == filtered.shape == terminal.shape and raw.shape[1] == 3
    assert np.array_equal(np.sort(np.concatenate(sessions)), np.arange(len(raw)))
    for a in (raw, filtered):
        assert np.all(a >= 0) and np.allclose(a.sum(1), 1, atol=1e-6)
    assert np.isfinite(terminal).all()
    out = {k: np.zeros_like(terminal, dtype=np.float64) for k in ('prior', 'reliable', 'cyclic')}
    error = 0.
    for rows in sessions:
        seen = np.zeros(3, int)
        support = np.zeros(2)
        active = 0
        for t, row in enumerate(rows):
            if t >= 20 and t % 20 == 0:
                # select() keeps the top half of each pseudo-class, minimum three.
                selected = np.minimum(seen, np.maximum(3, (seen+1)//2))
                log_prior = np.log((selected+1)/(selected.sum()+3))
            seen[raw[row].argmax()] += 1
            w = .10 * np.clip((t-90+1)/120, 0., 1.)
            if w == 0:
                assert np.array_equal(terminal[row], filtered[row])
                continue
            original = np.log(np.clip(filtered[row], 1e-6, 1))
            weighted = terminal[row] - original
            q = np.exp(weighted/w)
            # Each clipped LDA probability can add at most 1e-6 to its unrounded sum.
            error = max(error, abs(1-q.sum()))
            assert np.isfinite(q).all() and np.all(q <= 1+1e-9) and np.all(q >= 1e-6-1e-10)
            assert abs(1-q.sum()) <= 3.01e-6
            assert np.allclose(original+w*np.log(q), terminal[row], rtol=0, atol=1e-12)
            change = np.clip(1+1.5*((20/3+support)/(20+active)-1/3), .5, 1.5)-1
            out['prior'][row] = -.5*w*(log_prior-log_prior.mean())
            out['reliable'][row] = change[0]*weighted
            out['cyclic'][row] = change[1]*weighted
            forecast = q.argmax()
            support += raw[row, [forecast, (forecast+1)%3]]
            active += 1
    out['stack'] = out['prior']+out['reliable']
    return out, error
