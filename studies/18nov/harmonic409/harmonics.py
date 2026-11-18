"""Within-derivation harmonic consistency at fixed frequencies, in official windows."""
import numpy as np


def features(window, pairs, verify=False):
    signal = window[pairs[:, 0]].astype(float) - window[pairs[:, 1]]
    frames = np.lib.stride_tricks.sliding_window_view(signal, 120, axis=1)[:, ::60]
    spectrum = np.fft.rfft((frames - frames.mean(-1, keepdims=True)) * np.hanning(120))
    power = np.log(np.maximum(1e-20, np.mean(abs(spectrum[:, :, 4:37])**2, axis=1)))
    bins = np.arange(4, 19)
    fundamental = spectrum[:, :, bins]**2
    harmonic = spectrum[:, :, 2 * bins]
    angles = np.angle(harmonic)
    rng = np.random.default_rng(409)
    # Nonidentity shift guarantees a derangement and preserves phase sequence.
    shifted = np.roll(angles, 3, axis=1)
    rotated = angles + rng.uniform(-np.pi, np.pi, size=angles.shape[1:])
    denom = np.sqrt(np.sum(abs(fundamental)**2, axis=1) * np.sum(abs(harmonic)**2, axis=1))
    out = []
    for term in (harmonic, abs(harmonic)*np.exp(1j*shifted), abs(harmonic)*np.exp(1j*rotated)):
        if verify:
            np.testing.assert_allclose(abs(term), abs(harmonic), rtol=1e-12, atol=1e-12)
        moment = np.sum(fundamental * term.conjugate(), axis=1)
        out.append((abs(moment) / np.maximum(denom, 1e-20)).ravel())
    return (power.ravel(), *out)
