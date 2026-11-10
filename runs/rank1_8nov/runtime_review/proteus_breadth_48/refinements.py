"""Controlled refinements to existing spectral and covariance features."""

import numpy as np
from scipy.signal.windows import hann
from scipy.special import ndtri
from scipy.stats import rankdata

from experiments.features import bandpass
from signal_views import hann_geometry, waveform_direction
from spectral import PAIRS, spectral


def append(base, block):
    return np.column_stack((base, block.reshape(len(base), -1)))


def paired_indices(channels):
    return np.array([[channels.index(ch) for ch in pair] for pair in PAIRS]).T


def lag_contrast(raw, channels):
    views = waveform_direction(raw, channels)
    base, plain = views["hann_real"], views["hann_even"]
    energy = plain[:, base.shape[1]:].reshape(len(raw), 4, len(channels))
    logged = np.log(energy.clip(1e-8))
    relative = logged - logged.mean(axis=1, keepdims=True)
    shuffle = np.argsort(np.random.default_rng(7705).random((len(raw), 4)), axis=1)
    scrambled = np.take_along_axis(relative, shuffle[..., None], axis=1)
    return {"hann": base, "plain": plain, "log_energy": append(base, logged),
            "relative": append(base, relative), "shuffled": append(base, scrambled)}


def hemispheres(raw, channels):
    left, right = paired_indices(channels)
    p, z = spectral(raw, hann(raw.shape[-1])[None], np.r_[left, left, right, left],
                    np.r_[right, np.roll(left, 1), np.roll(right, 1), np.roll(right, 7)])
    mirror, lh, rh, rewire = np.split(z.real, 4, axis=-1)
    base = p.reshape(len(raw), -1)
    return {"power": base, "mirror": append(base, mirror),
            "same_side": append(base, (lh + rh) * .5),
            "rewired": append(base, rewire)}


def correlation(x, left, right):
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / x.std(axis=-1, keepdims=True).clip(1e-8)
    return (x @ x.swapaxes(-1, -2))[:, left, right] / x.shape[-1]


def rank_features(raw, left, right, bands):
    linear, rank = [], []
    for lo, hi in bands:
        x = bandpass(raw, lo, hi)
        g = ndtri((rankdata(x, axis=-1) - .5) / x.shape[-1]).astype(np.float32)
        linear.append(correlation(x, left, right))
        rank.append(correlation(g, left, right))
    return np.column_stack(linear), np.column_stack(rank)


def compact_rank(raw, channels):
    left, right = paired_indices(channels)
    base = waveform_direction(raw, channels)["hann_even"]
    linear, rank = rank_features(raw, left, right, ((4, 8), (8, 13), (13, 25), (25, 40)))
    return {"hann_lag": base, "linear": append(base, linear), "rank": append(base, rank)}


def band_rank(raw, channels):
    # All sensor pairs. Compare two bands in each group with identical widths in feature space.
    left, right = np.triu_indices(len(channels), k=1)
    base = hann_geometry(raw, channels)["power"]
    lm, rm = rank_features(raw, left, right, ((8, 13), (13, 25)))
    lo, ro = rank_features(raw, left, right, ((4, 8), (25, 40)))
    return {"power": base, "mid_linear": append(base, lm), "mid_rank": append(base, rm),
            "outer_linear": append(base, lo), "outer_rank": append(base, ro)}
