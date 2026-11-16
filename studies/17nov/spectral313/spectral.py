"""Per-window sensor cross-spectral powers and signed coherency views."""

import numpy as np

BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (45, 58))
UPPER = np.triu_indices(41, 1)
FLOOR = 1e-12


def cross_features(frequency, taper_power):
    f = np.asarray(frequency, dtype=np.complex128)
    if (f.ndim != 3 or f.shape[1] != 41 or f.shape[2] < 2 or
            not np.isfinite(f).all() or not np.isfinite(taper_power) or taper_power <= 0):
        raise ValueError('expected finite 41-channel multi-bin Fourier coefficients')
    cross = np.matmul(f, f.conj().transpose(0, 2, 1)) / (taper_power * f.shape[2])
    diag = np.maximum(cross.real.diagonal(axis1=1, axis2=2), FLOOR)
    denom = np.sqrt(diag[:, UPPER[0]] * diag[:, UPPER[1]])
    pairs = cross[:, UPPER[0], UPPER[1]] / denom
    result = (np.log(diag), pairs.real, pairs.imag)
    if any(not np.isfinite(a).all() for a in result):
        raise ValueError('nonfinite cross-spectral features')
    return result


def extract(windows):
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (41, 480) or not len(x) or not np.isfinite(x).all():
        raise ValueError('expected finite nonempty N × 41 × 480 windows')
    x = x - x.mean(axis=1, keepdims=True)
    views = [[None] * 3 for _ in BANDS]
    for segment, (start, stop) in enumerate(((0, 480), (0, 240), (240, 480))):
        part = x[:, :, start:stop]
        part = part - part.mean(axis=-1, keepdims=True)
        taper = np.hanning(stop - start)
        fft = np.fft.rfft(part * taper, axis=-1)
        frequencies = np.fft.rfftfreq(stop - start, d=1/120)
        for band, (lo, hi) in enumerate(BANDS):
            mask = (frequencies >= lo) & (frequencies < hi)
            if mask.sum() < 2:
                raise ValueError('band has fewer than two spectral bins')
            views[band][segment] = cross_features(fft[:, :, mask], float(taper @ taper))
    return tuple(np.concatenate([views[b][s][kind] for b in range(7) for s in range(3)], axis=1)
                 for kind in range(3))
