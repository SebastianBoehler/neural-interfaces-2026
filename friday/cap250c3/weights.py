"""Discount inconsistent refit targets while preserving each class's loss mass."""
import numpy as np


def policy_weights(arm, first, current, weights):
    disagreement = first != current
    if arm == 'stock' or not disagreement.any():
        return weights, int(disagreement.sum())

    marks = disagreement.copy()
    if arm == 'placebo':
        for start in range(0, len(first), 20):
            rows = np.arange(start, min(start+20, len(first)))
            for label in np.unique(current[rows]):
                group = rows[current[rows] == label]
                marks[group] = np.roll(disagreement[group], 1)
                assert marks[group].sum() == disagreement[group].sum()
    elif arm != 'primary':
        raise ValueError(f'unknown refit treatment {arm}')

    adjusted = weights*np.where(marks, .5, 1.)
    for label in np.unique(current):
        group = current == label
        target = weights[group].sum()
        adjusted[group] *= target/adjusted[group].sum()
        np.testing.assert_allclose(adjusted[group].sum(), target, atol=1e-10, rtol=0)
    assert np.isfinite(adjusted).all() and (adjusted > 0).all()
    return adjusted, int(disagreement.sum())
