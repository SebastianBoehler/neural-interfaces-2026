"""Cache fixed per-window spectral views once; labels are never loaded."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke import digest  # noqa: E402
from spectral import extract  # noqa: E402


def main():
    data = Path(os.environ['S313_DATA']).resolve(strict=True)
    gate = json.loads(Path('runs/s313_smoke/report.json').read_text())
    if (gate['no_labels_read'] is not True or gate['phase_fixture_passed'] is not True or gate['data_sha256'] != digest(data) or
            gate['source_sha256'] != digest(Path('source.sha256')) or
            max(gate['reference_max_abs']) > 1e-9 or gate['order_max_abs'] != 0 or
            min(gate['real_imag_max_abs']) < 1e-8 or gate['projected_2655_seconds'] > 2400 or
            gate['peak_rss_kib'] >= 12*1024*1024 or len(gate['cpu_affinity']) < 6):
        raise ValueError('real-window spectral smoke is absent or mismatched')
    out = Path('runs/s313_features')
    out.mkdir(parents=True, exist_ok=False)
    with np.load(data, allow_pickle=False) as z:
        x = z['X'].copy()
    if x.shape != (2655, 41, 480) or not np.isfinite(x).all():
        raise ValueError('official model-facing input shape or values differ')
    shapes = {'power': (len(x), 861), 'real': (len(x), 17220), 'imag': (len(x), 17220)}
    arrays = {name: np.lib.format.open_memmap(out / f'{name}.npy', mode='w+', dtype=np.float32, shape=shape)
              for name, shape in shapes.items()}
    started = time.perf_counter()
    for start in range(0, len(x), 16):
        stop = min(start+16, len(x))
        values = extract(x[start:stop])
        for (name, target), source in zip(arrays.items(), values):
            if not np.isfinite(source).all():
                raise ValueError(f'nonfinite {name} at row {start}')
            target[start:stop] = source
    for target in arrays.values():
        target.flush()
    hashes = {name: digest(out / f'{name}.npy') for name in arrays}
    report = {'no_labels_read': True, 'data_sha256': digest(data), 'source_sha256': digest(Path('source.sha256')),
              'smoke_sha256': digest(Path('runs/s313_smoke/report.json')), 'files_sha256': hashes,
              'shapes': shapes, 'dtype': 'float32', 'batch': 16, 'seconds': time.perf_counter()-started}
    with (out / 'manifest.json').open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print('all fixed spectral views cached; no labels or accuracy read', flush=True)


if __name__ == '__main__':
    main()
