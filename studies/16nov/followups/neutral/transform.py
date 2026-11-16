"""Explicit fixed uniform-information control; never a prediction fallback."""
import numpy as np


def apply(p):
    p = np.asarray(p)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('invalid shape or nonfinite EEG stream')
    return np.full_like(p, 1. / 3.)
