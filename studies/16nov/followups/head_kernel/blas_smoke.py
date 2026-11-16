"""Validate explicit OpenBLAS on checked P01 source data before timing."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blas import Blas  # noqa: E402
from kernel import Kernel  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    old = Path(os.environ['S311_OLD'])
    matrix = old / 'runs/s311_smoke/source_matrix.npz'
    original = json.loads((old / 'runs/s311_smoke/report.json').read_text())
    if digest(matrix) != original['matrix_sha256'] or original['held_accuracy_computed'] is not False:
        raise ValueError('source matrix differs from verified P01 fixture')
    out = Path('runs/blas_smoke')
    out.mkdir(parents=True, exist_ok=False)
    with np.load(matrix, allow_pickle=False) as z:
        x, y, theta = (np.ascontiguousarray(z[k]) for k in ('x', 'y', 'theta'))
    reference, candidate = Kernel(os.environ['S311_LIB']), Blas(os.environ['S311_BLAS_LIB'])
    info = candidate.info()
    if ('libopenblasp.so.0' not in info['path'] or 'SINGLE_THREADED' in info['config'] or
            'USE64BITINT' in info['config'] or info['threads'] != 6):
        raise ValueError(f'unexpected BLAS library, ABI or thread count: {info}')
    checks = {}
    for name, xx, yy in (('first32', x[:32], y[:32]), ('full', x, y)):
        a, ga = reference.objective(xx, yy, theta, analytical=False)
        b, gb = candidate.objective(xx, yy, theta)
        difference, gradient = abs(a-b), float(np.max(np.abs(ga-gb)))
        if not np.isfinite([a, b, difference, gradient]).all() or max(difference, gradient) > 1e-9:
            raise ValueError(f'{name}: BLAS loss or gradient differs')
        checks[name] = {'loss': b, 'loss_abs_diff': difference, 'gradient_max_abs_diff': gradient}
    bad = x[:1].copy()
    bad[0, 0] = np.nan
    try:
        candidate.objective(bad, y[:1], theta)
    except RuntimeError as exc:
        if 'nonfinite' not in str(exc):
            raise
    else:
        raise ValueError('BLAS kernel accepted nonfinite input')
    report = {'checks': checks, 'no_held_accuracy': True, 'matrix_sha256': digest(matrix),
              'reference_library_sha256': digest(Path(os.environ['S311_LIB'])),
              'blas_library_sha256': digest(Path(os.environ['S311_BLAS_LIB'])),
              'source_manifest_sha256': digest(Path('source.sha256')), 'blas_info': info}
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print('real source mini/full OpenBLAS parity and NaN guard passed; no held score', flush=True)


if __name__ == '__main__':
    main()
