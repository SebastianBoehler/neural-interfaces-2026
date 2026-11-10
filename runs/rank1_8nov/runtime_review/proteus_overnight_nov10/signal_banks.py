"""Window-local feature banks for the overnight PROTEUS screens."""

import numpy as np
from scipy.signal.windows import hann
from scipy.special import ndtri
from scipy.stats import rankdata

from experiments.features import bandpass, logm, upper
from refinements import append, paired_indices
from signal_views import hann_geometry, moments, waveform_direction
from spectral import spectral

MID = ((8, 13), (13, 25))


def mid_matrices(raw, transforms):
    values = {key: [] for key in transforms}
    for lo, hi in MID:
        filtered = bandpass(raw, lo, hi)
        for name, transform in transforms.items():
            x = transform(filtered)
            x = x - x.mean(axis=-1, keepdims=True)
            x = x / x.std(axis=-1, keepdims=True).clip(1e-8)
            values[name].append(x @ x.swapaxes(-1, -2) / x.shape[-1])
    return {name: np.stack(block, axis=1) for name, block in values.items()}


def normal_ranks(x):
    return ndtri((rankdata(x, axis=-1) - .5) / x.shape[-1]).astype(np.float32)


def power(raw, channels):
    return hann_geometry(raw, channels)["power"]


def covariance_banks(raw, channels):
    banks = mid_matrices(raw, {"linear": lambda x: x, "rank": normal_ranks,
                                "bounded": lambda x: np.tanh(
                                    (x - x.mean(axis=-1, keepdims=True)) /
                                    (2 * x.std(axis=-1, keepdims=True).clip(1e-8)))})
    i, j = np.triu_indices(len(channels), 1)
    base = power(raw, channels)
    return {"power": base, **{name: append(base, block[..., i, j]) for name, block in banks.items()}}


def half_window(raw, channels):
    full = waveform_direction(raw, channels)
    base = full["hann_real"]
    halves = [moments(raw[..., t:t+240])[0] for t in (0, 240)]
    arithmetic = .5 * (halves[0] + halves[1])
    geometric = np.sqrt(halves[0] * halves[1])
    return {"hann": base, "full": full["hann_even"], "arithmetic": append(base, arithmetic),
            "geometric": append(base, geometric)}


def dense_spectral(raw, channels):
    left, right = np.triu_indices(len(channels), 1)
    p, coherency = spectral(raw, hann(480)[None], left, right)
    base = p.reshape(len(raw), -1)
    # Use spectral bands 8–13 and 13–20, with the identical boundaries for time-domain controls.
    linear, rank = [], []
    for lo, hi in ((8, 13), (13, 20)):
        x = bandpass(raw, lo, hi)
        linear.append(correlation_matrix(x, left, right))
        rank.append(correlation_matrix(normal_ranks(x), left, right))
    return {"power": base, "spectral": append(base, coherency[:, 1:3].real),
            "linear": append(base, np.column_stack(linear)), "rank": append(base, np.column_stack(rank))}


def correlation_matrix(x, left, right):
    x = x - x.mean(axis=-1, keepdims=True)
    x /= x.std(axis=-1, keepdims=True).clip(1e-8)
    return (x @ x.swapaxes(-1, -2))[:, left, right] / x.shape[-1]


def prototypes(raw, channels):
    eye = np.eye(len(channels))
    blocks = mid_matrices(raw, {"rank": normal_ranks, "linear": lambda x: x})
    return {name: upper(logm(.95 * block + .05 * eye)).reshape(len(raw), -1)
            for name, block in blocks.items()}


def tile_bank(raw, channels):
    pair_idx = paired_indices(channels).T
    regions = (pair_idx[[0, 8, 9], 0], pair_idx[[0, 8, 9], 1],
               pair_idx[[1, 6, 7], 0], pair_idx[[1, 6, 7], 1],
               pair_idx[[2, 3, 4, 5], 0], pair_idx[[2, 3, 4, 5], 1],
               pair_idx[[10, 11], 0], pair_idx[[10, 11], 1])
    centered = raw - raw.mean(axis=1, keepdims=True)
    tiles = []
    for lo, hi in ((4, 8), (8, 13), (13, 25), (25, 40)):
        x = bandpass(centered, lo, hi).reshape(len(raw), len(channels), 12, 40)
        energy = np.log(np.mean(x*x, axis=-1).clip(1e-8))
        energy -= energy.mean(axis=1, keepdims=True)
        tiles.extend(energy[:, indices].mean(axis=1) for indices in regions)
    return {"tiles": np.stack(tiles, axis=2)}


def cross_view(raw, channels):
    left, right = paired_indices(channels)
    matrices = mid_matrices(raw, {"rank": normal_ranks})["rank"]
    spectral_rank = matrices[..., left, right].reshape(len(raw), -1)
    lag = waveform_direction(raw, channels)["hann_even"]
    return {"rank": spectral_rank, "temporal": lag}
