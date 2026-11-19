"""Shared offline and streaming covariance transforms."""
import numpy as np


def center(x):
    return (x - x.mean(-1, keepdims=True)).clip(-250, 250)


def covariance(x):
    return x @ x.swapaxes(-1, -2) / x.shape[-1]


def matrix_power(matrix, exponent):
    scale = np.trace(matrix, axis1=-2, axis2=-1)[..., None] / matrix.shape[-1]
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, np.maximum(scale * 0.06, 1e-5))
    return (vectors * values[..., None, :] ** exponent) @ vectors.swapaxes(-1, -2)


def transform(cov):
    # Partial whitening preserves scalp topology for the pretrained spatial encoder.
    scale = np.trace(cov, axis1=-2, axis2=-1)[..., None, None] / cov.shape[-1]
    scale = np.maximum(scale, 1e-5)
    cov = cov / scale
    return ((0.65 * matrix_power(cov, -0.5) + 0.35 * np.eye(cov.shape[-1])) / np.sqrt(scale)).astype(np.float32)


def update_transform(total, count, source):
    return transform((total + 10 * source) / (count + 10))


def fingerprint(x, cov):
    """Class-blind spatial correlations and slow-to-fast relative band power."""
    rms = np.diagonal(cov, axis1=-2, axis2=-1).clip(1e-5) ** 0.5
    corr = cov / (rms[..., :, None] * rms[..., None, :])
    i, j = np.tril_indices(x.shape[-2], -1)
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    bands = np.array([power[..., lo:hi].sum(-1) for lo, hi in ((8, 16), (16, 32), (32, 52), (52, 80), (80, 125), (125, 180))])
    bands = np.moveaxis(bands, 0, -2)
    logpow = np.log(bands + 1e-5)
    logpow -= logpow.mean(axis=-1, keepdims=True)
    return np.concatenate((corr[..., i, j], logpow.reshape(*x.shape[:-2], -1) * 0.25), axis=-1)


def spectral_fingerprint(x, cov):
    """Band-specific scalp coherence, measured before any adaptive alignment."""
    spectrum = np.fft.rfft(x * np.hanning(x.shape[-1]), axis=-1).astype(np.complex64)
    i, j = np.tril_indices(x.shape[-2], -1)
    features = [fingerprint(x, cov)]
    for lo, hi in ((12, 28), (28, 48), (48, 80), (80, 132)):
        features_band = spectrum[..., lo:hi]
        cross = (features_band @ features_band.swapaxes(-1, -2).conj()).real
        rms = np.maximum(np.diagonal(cross, axis1=-2, axis2=-1), 1e-4) ** 0.5
        features.append((cross[..., i, j] / (rms[..., i] * rms[..., j])).astype(np.float32))
    return np.concatenate(features, axis=-1)
