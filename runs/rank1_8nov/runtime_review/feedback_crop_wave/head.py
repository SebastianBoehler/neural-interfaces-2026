"""Matched spectral descriptors for fixed early, late, and full EEG windows."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from electrode_drift_wave import head as spectral

CHANNEL_ORDER = spectral.CHANNEL_ORDER
FEATURE_DIM = spectral.FEATURE_DIM
fit = spectral.fit
logits = spectral.logits

CROPS = {"pre": slice(0, 180), "post": slice(300, 480), "full": slice(0, 480)}
NFFT = 480


def _taper(n_samples):
    window = np.hanning(n_samples)
    return window / np.linalg.norm(window)


_TAPERS = {name: _taper(crop.stop - crop.start) for name, crop in CROPS.items()}


def _block(raw, taper):
    centered = raw - raw.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered * taper, n=NFFT, axis=-1)
    powers = np.abs(spectrum) ** 2
    floor = np.maximum(powers.max(axis=(1, 2))[:, None] * 1e-10, np.finfo(np.float64).tiny)
    left, right = spectral._LEFT, spectral._RIGHT
    columns = []
    for band in spectral._BANDS:
        power = np.take(powers, band, axis=-1).mean(axis=-1) + floor
        log_power = np.log(power)
        regional = np.column_stack([np.median(log_power[:, group], axis=1)
                                    for group in spectral._REGION_INDEX])
        coeff = np.take(spectrum, band, axis=-1)
        cov = (coeff[:, left] * coeff[:, right].conj()).real.mean(axis=-1)
        shrunk_corr = 0.9 * cov / (np.sqrt(power[:, left]) * np.sqrt(power[:, right]))
        columns.extend((regional, shrunk_corr))
    return np.concatenate(columns, axis=1)


def features(X):
    """Compute the same 92 features on each fixed crop, with no row dependencies."""
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[None]
    if X.ndim != 3 or X.shape[1:] != (len(CHANNEL_ORDER), NFFT) or X.dtype.kind != "f":
        raise ValueError("Expected floating-point EEG shaped (N, 41, 480) in CHANNEL_ORDER")
    result = {name: np.empty((len(X), FEATURE_DIM), dtype=np.float64) for name in CROPS}
    for start in range(0, len(X), 32):
        rows = slice(start, start + 32)
        block = X[rows].astype(np.float64, copy=False)
        if not np.isfinite(block).all():
            raise ValueError("EEG windows must be finite")
        for name, crop in CROPS.items():
            result[name][rows] = _block(block[..., crop], _TAPERS[name])
    if any(not np.isfinite(view).all() for view in result.values()):
        raise ValueError("Crop features must be finite")
    return result
