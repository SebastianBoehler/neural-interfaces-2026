"""New source-only signal and readout hypotheses."""

import numpy as np
from scipy.fft import irfft, rfft
from scipy.signal.windows import hann
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler

from experiments.features import bandpass
from refinements import append
from signal_views import hann_geometry, waveform_direction
from spectral import fit_head


def phase_control(x, seed):
    z = rfft(x, axis=-1)
    theta = np.random.default_rng(seed).uniform(-np.pi, np.pi, size=(len(x), 1, z.shape[-1]))
    theta[..., (0, -1)] = 0
    return irfft(z * np.exp(1j * theta), n=x.shape[-1], axis=-1).astype(np.float32)


def ordinal_bank(x):
    blocks = []
    for d in (3, 8, 16):
        a, b, c = x[..., :-2*d], x[..., d:-d], x[..., 2*d:]
        # Use temporal position to break exact ties.
        code = (a <= b).astype(np.uint8) + 2 * (a <= c) + 4 * (b <= c)
        blocks.extend([(code == j).mean(axis=-1) for j in (0, 1, 3, 4, 6, 7)])
    return np.concatenate(blocks, axis=1)


def ordinal(raw, channels):
    base = waveform_direction(raw, channels)["hann_even"]
    x = raw - raw.mean(axis=1, keepdims=True)
    return {"hann_lag": base, "ordinal": append(base, ordinal_bank(x)),
            "phase_scrambled": append(base, ordinal_bank(phase_control(x, 7319)))}


def bispectral_bank(x):
    n = len(x)
    tiles = np.lib.stride_tricks.sliding_window_view(x, 120, axis=-1)[..., ::60, :]
    z = rfft(tiles * hann(120).astype(np.float32), axis=-1)
    blocks = []
    for a, b in ((np.arange(4, 8), np.arange(8, 13)), (np.arange(8, 13), np.arange(8, 13)),
                 (np.arange(8, 13), np.arange(13, 20))):
        prod = z[..., a][..., :, None] * z[..., b][..., None, :]
        target = z[..., a[:, None] + b]
        axis = (-3, -2, -1)
        num = np.mean(prod * target.conj(), axis=axis)
        den = np.sqrt(np.mean(np.abs(prod)**2, axis=axis) * np.mean(np.abs(target)**2, axis=axis))
        coupling = num / den.clip(1e-12)
        blocks.extend((coupling.real, coupling.imag))
    return np.column_stack(blocks).reshape(n, -1)


def bispectrum(raw, channels):
    base = hann_geometry(raw, channels)["mirror_all"]
    x = raw - raw.mean(axis=1, keepdims=True)
    return {"hann": base, "bispectrum": append(base, bispectral_bank(x)),
            "phase_scrambled": append(base, bispectral_bank(phase_control(x, 1531)))}


def spatial_modes(raw, channels):
    base = hann_geometry(raw, channels)["mirror_all"]
    spread, topography, placebo = [], [], []
    rng = np.random.default_rng(1221)
    order = np.argsort(rng.random((len(raw), len(channels))), axis=1)
    for low, high in ((4, 8), (8, 13), (13, 25), (25, 40)):
        x = bandpass(raw - raw.mean(axis=1, keepdims=True), low, high)
        cov = x @ x.swapaxes(-1, -2)
        values, vectors = np.linalg.eigh(cov)
        shares = values[:, -3:] / values.sum(axis=1, keepdims=True).clip(1e-12)
        participation = np.sum(vectors[:, :, -3:]**2 * shares[:, None, :], axis=-1)
        spread.append(shares)
        topography.append(participation)
        placebo.append(np.take_along_axis(participation, order, axis=1))
    global_view = append(base, np.column_stack(spread))
    return {"hann": base, "eigenvalues": global_view,
            "eigenmodes": append(global_view, np.column_stack(topography)),
            "rewired": append(global_view, np.column_stack(placebo))}


def interaction_bank(raw, channels):
    return {"hann_lag": waveform_direction(raw, channels)["hann_even"]}


def interaction_forecasts(views, fit_rows, held, labels, seed, fold):
    x = views["hann_lag"]
    scaler = StandardScaler().fit(x[fit_rows])
    t, h = scaler.transform(x[fit_rows]), scaler.transform(x[held])
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 6422]))
    projection = rng.normal(size=(t.shape[1], 512)) / np.sqrt(t.shape[1])
    bias = rng.uniform(0, 2 * np.pi, size=projection.shape[1])
    t_projection, h_projection = t @ projection, h @ projection
    features = {"hann_lag": (t, h), "linear_projection": (t_projection, h_projection),
                "nonlinear": (np.cos(t_projection + bias), np.cos(h_projection + bias))}
    return {arm: softmax(fit_head(a, labels[fit_rows]).decision_function(b), axis=1)
            for arm, (a, b) in features.items()}
