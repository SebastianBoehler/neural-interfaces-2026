"""Paired warm CPU objective and same-settings full-solve benchmark."""

import hashlib
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
from kernel import Kernel  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def compare(lib, x, y, theta):
    a, ga = lib.objective(x, y, theta, analytical=False)
    b, gb = lib.objective(x, y, theta, analytical=True)
    delta, grad = abs(a - b), float(np.max(np.abs(ga - gb)))
    if not np.isfinite([a, b, delta, grad]).all() or max(delta, grad) > 1e-9:
        raise ValueError('fixed-parameter analytical identity failed')
    return {'loss_abs_diff': delta, 'gradient_max_abs_diff': grad}


def main():
    report_path = Path('runs/s311_smoke/report.json')
    matrix_path = report_path.parent / 'source_matrix.npz'
    smoke = json.loads(report_path.read_text())
    if (smoke['held_accuracy_computed'] is not False or digest(matrix_path) != smoke['matrix_sha256'] or
            digest(Path(os.environ['S311_LIB'])) != smoke['library_sha256'] or
            digest(Path('source.sha256')) != smoke['source_manifest_sha256']):
        raise ValueError('source-only real-data smoke or hashes differ')
    target = Path('runs/s311_bench/report.json')
    if target.exists() or target.parent.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True)
    with np.load(matrix_path, allow_pickle=False) as z:
        x, y, theta = (np.ascontiguousarray(z[k]) for k in ('x', 'y', 'theta'))
    if list(x.shape) != smoke['shape'] or not x.flags.c_contiguous:
        raise ValueError('matrix layout differs from the checked source fixture')
    lib = Kernel(os.environ['S311_LIB'])
    fixed_check = compare(lib, x, y, theta)
    timings = []
    for order in ((False, True), (True, False), (False, True)):
        for analytical in order:
            start = time.perf_counter()
            loss, grad = lib.objective(x, y, theta, analytical)
            seconds = time.perf_counter() - start
            if not np.isfinite(loss) or not np.isfinite(grad).all():
                raise ValueError('nonfinite timed objective')
            timings.append({'method': 'analytical' if analytical else 'autograd', 'seconds': seconds,
                            'objective': loss, 'max_gradient': float(np.max(np.abs(grad)))})
    fits = []
    objective_target = smoke['checks']['full']['reference_loss'] + 1e-5
    for analytical in (False, True):
        start = time.perf_counter()
        fitted, loss, grad, calls = lib.fit(x, y, analytical)
        elapsed = time.perf_counter() - start
        check = compare(lib, x, y, fitted)
        reference_loss, reference_gradient = lib.objective(x, y, fitted, analytical=False)
        if (not np.isfinite([loss, grad, elapsed, reference_loss]).all() or calls <= 0 or
                loss > objective_target or grad > 1e-5 or abs(loss - reference_loss) > 1e-9):
            raise ValueError('fit did not reach common target and convergence guard')
        fits.append({'method': 'analytical' if analytical else 'autograd', 'seconds': elapsed,
                     'objective': loss, 'max_gradient': grad, 'objective_evaluations': calls,
                     'reference_gradient_max': float(np.max(np.abs(reference_gradient))),
                     'time_to_target_upper_bound_s': elapsed, 'fixed_check': check})
        with (target.parent / 'fit_progress.jsonl').open('a') as f:
            f.write(json.dumps(fits[-1], sort_keys=True) + '\n')
    if abs(fits[0]['objective'] - fits[1]['objective']) > 1e-5:
        raise ValueError('full-fit objectives differ beyond matched-quality bound')
    with open('/proc/self/maps') as f:
        mapped = sorted({line.split()[-1] for line in f if '/' in line and any(k in line.lower() for k in ('torch', 'mkl', 'blas', 'gomp', 'iomp'))})
    output = {'no_held_accuracy': True, 'source_manifest_sha256': smoke['source_manifest_sha256'],
              'matrix_sha256': smoke['matrix_sha256'], 'library_sha256': smoke['library_sha256'],
              'source_shape': list(x.shape), 'source_layout': 'C-contiguous float64, input borrowed without copy',
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'platform': platform.platform(),
              'torch_threads': torch.get_num_threads(), 'threadpools': threadpool_info(), 'mapped_libraries': mapped,
              'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
              'torch_config': torch.__config__.show(), 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'fixed_parameter_check': fixed_check, 'objective_target': objective_target,
              'paired_warm_calls': timings, 'full_fits': fits,
              'time_limit_note': 'full solve times upper-bound first reaching target; per-iteration target time was not recorded'}
    with target.open('x') as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write('\n')
    print('paired source-only objective and full-solve benchmark completed; no held score', flush=True)


if __name__ == '__main__':
    main()
