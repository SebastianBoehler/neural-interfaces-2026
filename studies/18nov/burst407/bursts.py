"""Bipolar analytic-amplitude excursions and marginal-preserving controls."""
import numpy as np

BANDS = ((8, 13), (13, 21), (21, 33))
RNG_SEED = 407


def durations(mask):
    result = []
    for row in mask:
        edges = np.diff(np.r_[0, row.astype(int), 0])
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        longest = lengths.max(initial=0)
        sustained = lengths[lengths >= 12].sum() / max(1, lengths.sum())
        result.extend((longest / 120, sustained))
    return np.array(result)


def features(window, pairs, verify=False):
    x = window[pairs[:, 0]].astype(float) - window[pairs[:, 1]]
    padded = np.pad(x, ((0, 0), (120, 120)), mode='reflect')
    ft = np.fft.fft(padded)
    freq = np.fft.fftfreq(padded.shape[-1], 1 / 120)
    amp = np.concatenate([abs(np.fft.ifft(2 * ft * ((freq >= lo) & (freq < hi))))[:, 120:-120]
                          for lo, hi in BANDS])
    quantiles = np.quantile(amp, [.25, .5, .75, .90], axis=1)
    amplitude = np.log(np.maximum(np.vstack((np.mean(amp**2, axis=1)**.5, quantiles)), 1e-12)).ravel()
    elevated = amp > quantiles[2, :, None]
    rng = np.random.default_rng(RNG_SEED)
    sample = rng.permutation(480)
    block = np.arange(480).reshape(40, 12)[rng.permutation(40)].ravel()
    if verify:
        for order in (sample, block):
            np.testing.assert_array_equal(np.sort(amp[:, order]), np.sort(amp))
        np.testing.assert_array_equal(np.diff(block.reshape(40, 12)), 1)
    return amplitude, durations(elevated), durations(elevated[:, sample]), durations(elevated[:, block])
