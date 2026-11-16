"""Fixed current-row confidence gate; no normalization or clipping."""
import numpy as np


def apply(base, added, threshold):
    base, added = np.asarray(base), np.asarray(added)
    if base.ndim != 2 or base.shape[1] != 3 or added.shape != base.shape:
        raise ValueError('expected aligned three-class rows')
    if threshold not in (.6, .8) or not np.isfinite(base).all() or not np.isfinite(added).all():
        raise ValueError('non-finite mix or unregistered threshold')
    # Tolerance permits floating roundoff only; values remain unmodified.
    if np.any(base < -1e-7) or np.any(base > 1 + 1e-7) or not np.allclose(base.sum(1), 1, rtol=0, atol=1e-6):
        raise ValueError('unchanged outer mix is not a probability distribution')
    eligible = base.max(axis=1) <= threshold
    changed = np.any(base != added, axis=1) & eligible
    if not eligible.any() or not changed.any():
        raise ValueError('gate has no eligible changed row')
    result = np.where(eligible[:, None], added, base)
    if not np.array_equal(result[~eligible], base[~eligible]):
        raise ValueError('gate changed an ineligible row')
    return result, {'total': len(base), 'eligible': int(eligible.sum()), 'changed': int(changed.sum()),
                    'ineligible_unchanged': int((~eligible).sum()), 'threshold': threshold}
