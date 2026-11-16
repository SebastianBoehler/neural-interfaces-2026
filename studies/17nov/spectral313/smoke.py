"""Real-window spectral identity, causality and throughput smoke; no labels."""

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral import BANDS, FLOOR, UPPER, cross_features, extract  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def reference(window):
    x = window.astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    result = [[None]*3 for _ in BANDS]
    for segment, (start, stop) in enumerate(((0, 480), (0, 240), (240, 480))):
        part = x[:, start:stop]
        part = part - part.mean(axis=1, keepdims=True)
        taper = np.hanning(stop-start)
        frequencies = np.fft.rfftfreq(stop-start, d=1/120)
        f = np.fft.rfft(part*taper, axis=1)
        for band, (lo, hi) in enumerate(BANDS):
            indices = np.flatnonzero((frequencies >= lo) & (frequencies < hi))
            diag = np.maximum(np.array([np.sum(np.abs(row[indices])**2) for row in f]) /
                              (len(indices) * float(taper @ taper)), FLOOR)
            pair = np.array([np.sum(f[i, indices] * f[j, indices].conj()) /
                             (len(indices) * float(taper @ taper) * np.sqrt(diag[i]*diag[j]))
                             for i, j in zip(*UPPER)])
            result[band][segment] = np.log(diag), pair.real, pair.imag
    return tuple(np.concatenate([result[b][s][k] for b in range(7) for s in range(3)]) for k in range(3))


def main():
    data = Path(os.environ['S313_DATA']).resolve(strict=True)
    out = Path('runs/s313_smoke')
    out.mkdir(parents=True, exist_ok=False)
    fixture = np.zeros((1, 41, 6), dtype=np.complex128)
    fixture[:, 0] = 1
    fixture[:, 1] = 1j
    power, real, imag = cross_features(fixture, 1.0)
    opposite = fixture.copy()
    opposite[:, 1] *= -1
    p2, r2, i2 = cross_features(opposite, 1.0)
    global_shift = cross_features(fixture * np.exp(.7j), 1.0)
    zero_lag = fixture.copy()
    zero_lag[:, 1] = 1
    if (not np.array_equal(power, p2) or not np.array_equal(real, r2) or
            not np.array_equal(imag, -i2) or np.max(np.abs(imag)) < .9 or
            np.any(cross_features(zero_lag, 1.0)[2]) or
            any(not np.allclose(a, b, atol=1e-12, rtol=0) for a,b in zip((power, real, imag), global_shift))):
        raise ValueError('phase-lag, zero-lag or common Fourier-phase fixture failed')
    with np.load(data, allow_pickle=False) as z:
        windows = z['X'][:8].copy()
    start = time.perf_counter()
    result = extract(windows)
    elapsed = time.perf_counter()-start
    ref = reference(windows[0])
    errors = [float(np.max(np.abs(ref[k]-result[k][0]))) for k in range(3)]
    reverse = extract(windows[::-1])
    order_error = max(float(np.max(np.abs(a-b[::-1]))) for a,b in zip(result,reverse))
    active = [float(np.max(np.abs(result[k]))) for k in (1,2)]
    if max(errors) > 1e-9 or order_error != 0 or min(active) < 1e-8:
        raise ValueError('real-window spectral reference, statelessness or activity failed')
    if any(np.max(np.abs(a)) > 1+1e-10 for a in result[1:]):
        raise ValueError('cross-spectral normalization exceeds unit magnitude')
    report = {'no_labels_read': True, 'data_sha256': digest(data), 'source_sha256': digest(Path('source.sha256')),
              'windows': len(windows), 'shapes': [list(x.shape) for x in result], 'reference_max_abs': errors,
              'phase_fixture_passed': True,
              'order_max_abs': order_error, 'real_imag_max_abs': active, 'elapsed_seconds': elapsed,
              'projected_2655_seconds': elapsed*2655/len(windows), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print('real-window spectral reference and statelessness passed; no labels read', flush=True)


if __name__ == '__main__':
    main()
