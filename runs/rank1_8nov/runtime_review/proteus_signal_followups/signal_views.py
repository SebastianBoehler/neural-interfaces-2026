"""Fixed feature banks for two participant-disjoint EEG experiments."""

import numpy as np
from scipy.signal.windows import hann

from spectral import PAIRS, spectral

LAGS = (3, 8, 16, 30)


def hann_geometry(raw, channels):
    left, right = np.array([[channels.index(ch) for ch in pair] for pair in PAIRS]).T
    rewired = (np.roll(right, 3), np.roll(right, 7))
    p, z = spectral(raw, hann(raw.shape[-1])[None], np.tile(left, 3),
                    np.concatenate((right, *rewired)))
    n = len(raw)
    power = p.reshape(n, -1)
    views = {"power": power}
    for geometry, block in zip(("mirror", "rewired3", "rewired7"), np.split(z.real, 3, axis=-1)):
        for band, indices in (("all", slice(None)), ("mid", slice(1, 3)), ("high", slice(3, 5))):
            views[f"{geometry}_{band}"] = np.column_stack((power, block[:, indices].reshape(n, -1)))
    return views


def moments(raw):
    x = raw.copy()
    x -= x.mean(axis=1, keepdims=True)
    x -= x.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True))
    x = np.tanh(x / np.maximum(rms, 1e-6))
    even, odd = [], []
    for lag in LAGS:
        delta = x[..., lag:] - x[..., :-lag]
        square = delta * delta
        even.append(np.mean(square, axis=-1))
        odd.append(np.mean(square * delta, axis=-1))
    return np.concatenate(even, axis=1), np.concatenate(odd, axis=1)


def waveform_direction(raw, channels):
    left, right = np.array([[channels.index(ch) for ch in pair] for pair in PAIRS]).T
    p, z = spectral(raw, hann(raw.shape[-1])[None], left, right)
    power = p.reshape(len(raw), -1)
    real = z.real.reshape(len(raw), -1)
    even, odd = moments(raw)
    direction = np.random.default_rng(19462).choice((-1, 1), size=(len(raw), 1))
    return {"power": power,
            "hann_real": np.column_stack((power, real)),
            "even": np.column_stack((power, even)),
            "time_direction": np.column_stack((power, even, odd)),
            "random_reversal": np.column_stack((power, even, odd * direction)),
            "hann_even": np.column_stack((power, real, even)),
            "hann_direction": np.column_stack((power, real, even, odd)),
            "hann_random_reversal": np.column_stack((power, real, even, odd * direction))}
