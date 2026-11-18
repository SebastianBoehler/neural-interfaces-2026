"""Oriented phase slope from repeated Fourier frames in a single window."""
import hashlib

import numpy as np

PAIRS = (('C3','CP3'), ('C4','CP4'), ('Cz','CPz'), ('FC3','C3'),
         ('FC4','C4'), ('FCz','Cz'), ('F3','Fz'), ('F4','Fz'),
         ('F5','FC5'), ('F6','FC6'), ('P3','Pz'), ('P4','Pz'))
BANDS = ((4,8), (8,13), (13,21), (21,33))


def features(window, pairs, verify=False):
    x = window.astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    frames = np.lib.stride_tricks.sliding_window_view(x, 120, axis=-1)[:,::60]
    spectrum = np.fft.rfft(frames * np.hanning(120))
    power = np.mean(abs(spectrum)**2, axis=1)
    a, b = pairs.T
    cross = np.mean(spectrum[a].conj()*spectrum[b], axis=1)
    coherency = cross / np.sqrt(np.maximum(power[a]*power[b], 1e-24))
    # Derived only from this input. Duplicate windows get identical surrogates.
    seed = int.from_bytes(hashlib.sha256(window.tobytes()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    phase = np.exp(2j*np.pi*rng.random(power.shape))
    null = coherency * phase[a].conj() * phase[b]
    if verify:
        np.testing.assert_allclose(abs(null), abs(coherency), atol=1e-12)

    def slope(c):
        return (c[:,:-1].conj()*c[:,1:]).imag.mean(axis=1)

    out = [[], [], [], []]
    for lo, hi in BANDS:
        c = coherency[:,lo:hi]
        out[0].extend(np.log(np.maximum(power[:,lo:hi].mean(1), 1e-12)))
        out[1].extend(c.imag.mean(1))
        out[2].extend(slope(c))
        out[3].extend(slope(null[:,lo:hi]))
    return tuple(np.asarray(v) for v in out)
