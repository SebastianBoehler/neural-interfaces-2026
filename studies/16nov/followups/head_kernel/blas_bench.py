"""Paired source-only benchmark of original autograd and explicit LP64 BLAS."""

import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_info

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blas import Blas  # noqa: E402
from blas_smoke import digest  # noqa: E402
from kernel import Kernel  # noqa: E402


def compare(ref, raw, x, y, theta):
    a, ga = ref.objective(x, y, theta, analytical=False)
    b, gb = raw.objective(x, y, theta)
    loss, grad = abs(a-b), float(np.max(np.abs(ga-gb)))
    if not np.isfinite([a, b, loss, grad]).all() or max(loss, grad) > 1e-9:
        raise ValueError('BLAS objective/gradient identity failed')
    return {'loss_abs_diff': loss, 'gradient_max_abs_diff': grad}


def main():
    smoke = json.loads(Path('runs/blas_smoke/report.json').read_text())
    matrix = Path(os.environ['S311_OLD']) / 'runs/s311_smoke/source_matrix.npz'
    if (smoke['no_held_accuracy'] is not True or digest(matrix) != smoke['matrix_sha256'] or
            digest(Path(os.environ['S311_LIB'])) != smoke['reference_library_sha256'] or
            digest(Path(os.environ['S311_BLAS_LIB'])) != smoke['blas_library_sha256'] or
            digest(Path('source.sha256')) != smoke['source_manifest_sha256']):
        raise ValueError('checked BLAS smoke provenance differs')
    output = Path('runs/blas_bench')
    output.mkdir(parents=True, exist_ok=False)
    with np.load(matrix, allow_pickle=False) as z:
        x, y, theta = (np.ascontiguousarray(z[k]) for k in ('x', 'y', 'theta'))
    ref, raw = Kernel(os.environ['S311_LIB']), Blas(os.environ['S311_BLAS_LIB'])
    identity = compare(ref, raw, x, y, theta)
    timings = []
    for order in (('autograd', 'blas'), ('blas', 'autograd'), ('autograd', 'blas')):
        for method in order:
            start = time.perf_counter()
            loss, grad = ref.objective(x, y, theta, False) if method == 'autograd' else raw.objective(x, y, theta)
            timings.append({'method': method, 'seconds': time.perf_counter()-start, 'loss': loss,
                            'max_gradient': float(np.max(np.abs(grad)))})
    baseline = json.loads((Path(os.environ['S311_OLD']) / 'runs/s311_smoke/report.json').read_text())
    target = baseline['checks']['full']['reference_loss'] + 1e-5
    fits = []
    for method in ('autograd', 'blas'):
        start = time.perf_counter()
        fitted, loss, grad, calls = ref.fit(x, y, False) if method == 'autograd' else raw.fit(x, y)
        elapsed = time.perf_counter()-start
        check = compare(ref, raw, x, y, fitted)
        if not np.isfinite([loss, grad, elapsed]).all() or loss > target or grad > 1e-5 or calls <= 0:
            raise ValueError(f'{method}: full fit missed common quality target')
        fits.append({'method': method, 'seconds': elapsed, 'loss': loss, 'max_gradient': grad,
                     'objective_evaluations': calls, 'time_to_target_upper_bound_s': elapsed, 'identity': check})
        with (output / 'fit_progress.jsonl').open('a') as f:
            f.write(json.dumps(fits[-1], sort_keys=True) + '\n')
    if abs(fits[0]['loss']-fits[1]['loss']) > 1e-5:
        raise ValueError('full-fit objective gap exceeds registered common-quality bound')
    report = {'no_held_accuracy': True, 'source_manifest_sha256': smoke['source_manifest_sha256'],
              'matrix_sha256': smoke['matrix_sha256'], 'blas_library_sha256': smoke['blas_library_sha256'],
              'source_layout': 'C-contiguous float64, input borrowed without copy',
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'platform': platform.platform(),
              'torch_threads': torch.get_num_threads(), 'threadpools': threadpool_info(), 'blas_info': raw.info(),
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'fixed_identity': identity, 'objective_target': target, 'paired_warm_calls': timings,
              'full_fits': fits, 'time_note': 'full solve is an upper bound on reaching target'}
    with (output / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')
    print('paired autograd/LP64-BLAS benchmark complete; no held accuracy', flush=True)


if __name__ == '__main__':
    main()
