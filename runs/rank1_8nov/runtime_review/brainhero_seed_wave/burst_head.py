"""Decode trial-level EEG power distributions without retaining tile order."""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CHANNELS = 41
SAMPLES = 480
TILE = 120
STRIDE = 60
BANDS = ((4, 8), (8, 13), (13, 20), (20, 32), (32, 45))
HANN = np.hanning(TILE)
WIDTH = CHANNELS * len(BANDS)
BLOCK = 96
C = 0.1


def features(X):
    """Return median topography (N×205) and upper/lower spread (N×410)."""
    raw = np.asarray(X)
    if (raw.ndim != 3 or raw.shape[1:] != (CHANNELS, SAMPLES) or not len(raw)
            or not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all()):
        raise ValueError("Expected finite floating EEG windows with shape (N, 41, 480)")

    stationary = np.empty((len(raw), WIDTH))
    distribution = np.empty((len(raw), 2 * WIDTH))
    for start in range(0, len(raw), BLOCK):
        stop = min(start + BLOCK, len(raw))
        block = raw[start:stop].astype(np.float64, copy=False)
        tiles = sliding_window_view(block, TILE, axis=-1)[:, :, ::STRIDE, :]
        tapered = (tiles - tiles.mean(axis=-1, keepdims=True)) * HANN
        power = np.abs(np.fft.rfft(tapered, axis=-1)) ** 2
        band_power = np.stack([power[..., lo:hi].mean(axis=-1) for lo, hi in BANDS], axis=-1)
        # Common gain at any tile and band is removed in both arms.
        floor = np.maximum(1e-8 * band_power.mean(axis=1, keepdims=True), np.finfo(np.float64).tiny)
        log_power = np.log(np.maximum(band_power, floor))
        relative = log_power - log_power[:, :1]
        spatial = relative - relative.mean(axis=1, keepdims=True)
        ranks = np.sort(spatial, axis=2)
        median = ranks[:, :, 3, :]
        upper, lower = ranks[:, :, 5, :] - median, median - ranks[:, :, 1, :]
        stationary[start:stop] = median.reshape(len(block), WIDTH)
        distribution[start:stop, :WIDTH] = upper.reshape(len(block), WIDTH)
        distribution[start:stop, WIDTH:] = lower.reshape(len(block), WIDTH)
    if not np.isfinite(stationary).all() or not np.isfinite(distribution).all():
        raise ValueError("Band power overflowed during EEG feature extraction")
    return stationary, distribution


def _rows(rows, n):
    rows = np.asarray(rows)
    if (rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows < 0) or np.any(rows >= n) or len(np.unique(rows)) != len(rows)):
        raise ValueError("Expected nonempty unique row-relative EEG indices")
    return rows


@dataclass(frozen=True)
class BurstHeads:
    stationary_scaler: StandardScaler
    distribution_scaler: StandardScaler
    control: LogisticRegression
    full: LogisticRegression


def _matrices(stationary, distribution):
    stationary, distribution = map(np.asarray, (stationary, distribution))
    if (stationary.ndim != 2 or stationary.shape[1] != WIDTH or not len(stationary)
            or distribution.shape != (len(stationary), 2 * WIDTH)):
        raise ValueError("Expected matching stationary and spread EEG feature matrices")
    return stationary, distribution


def fit_heads(stationary, distribution, y, train):
    """Fit both heads and scalers using only the designated source rows."""
    stationary, distribution = _matrices(stationary, distribution)
    labels = np.asarray(y)
    if labels.shape != (len(stationary),):
        raise ValueError("Expected one class label per EEG feature row")
    train = _rows(train, len(stationary))
    source_labels = labels[train]
    if not np.array_equal(np.unique(source_labels), np.arange(3)):
        raise ValueError("Source EEG must contain classes 0, 1 and 2")
    x, z = stationary[train], distribution[train]
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("Source EEG features must be finite")
    stationary_scaler = StandardScaler().fit(x)
    distribution_scaler = StandardScaler().fit(z)
    x, z = stationary_scaler.transform(x), distribution_scaler.transform(z)
    control = LogisticRegression(C=C, max_iter=2000).fit(x, source_labels)
    full = LogisticRegression(C=C, max_iter=2000).fit(np.column_stack((x, z)), source_labels)
    return BurstHeads(stationary_scaler, distribution_scaler, control, full)


def predict_logits(heads, stationary, distribution):
    """Return control and full three-class logits from frozen source heads."""
    stationary, distribution = _matrices(stationary, distribution)
    if not np.isfinite(stationary).all() or not np.isfinite(distribution).all():
        raise ValueError("Prediction EEG features must be finite")
    x = heads.stationary_scaler.transform(stationary)
    z = heads.distribution_scaler.transform(distribution)
    return heads.control.decision_function(x), heads.full.decision_function(np.column_stack((x, z)))


def fit_logits(stationary, distribution, y, train, predict):
    """Fit on source rows and return the control and full held-row logits."""
    stationary, distribution = _matrices(stationary, distribution)
    train, predict = _rows(train, len(stationary)), _rows(predict, len(stationary))
    if np.intersect1d(train, predict).size:
        raise ValueError("Source and prediction rows must be disjoint")
    heads = fit_heads(stationary, distribution, y, train)
    return predict_logits(heads, stationary[predict], distribution[predict])
