"""Independent value/gradient fixtures and strict source/held output qualification."""
import ctypes
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from bindings import load, sha, MATRIX_SHA
from reference import ReferenceSpan, FullReference
from verify import objective, parity
from held import compare
from sklearn_reference import fit as sklearn_fit


def fixed(model, old, x, y):
    handle, stats = ctypes.c_void_p(), np.empty(7, np.float64)
    model.check(getattr(model.lib, model.create_symbol)(x.ctypes.data, y.ctypes.data, *x.shape,
                .3, ctypes.byref(handle), stats.ctypes.data))
    model.lib.s343r_gradient_vector.argtypes = [ctypes.c_void_p]*3
    checks = []
    try:
        for scale in (0., 1e-4):
            reduced = np.random.default_rng(343).normal(size=3*(int(stats[1])+1))*scale
            loss, grad = np.empty(1, np.float64), np.empty_like(reduced)
            theta, fullgrad = np.empty((3, x.shape[1]+1), np.float64), np.empty((3, x.shape[1]+1), np.float64)
            model.check(model.lib.s343r_objective(handle, reduced.ctypes.data, loss.ctypes.data, grad.ctypes.data))
            model.check(model.lib.s343r_reconstruct(handle, reduced.ctypes.data, theta.ctypes.data))
            model.check(model.lib.s343r_gradient_vector(handle, grad.ctypes.data, fullgrad.ctypes.data))
            expected, expected_grad = old.objective(x, y, theta, False)
            delta, error = abs(expected-loss[0]), float(abs(expected_grad-fullgrad).max())
            if max(delta, error) > 1e-9:
                raise ValueError(f'independent value/gradient mismatch {delta=} {error=}')
            checks.append(dict(scale=scale, objective_error=delta, gradient_error=error))
    finally:
        model.lib.s343r_destroy(handle)
    return checks


def main():
    old, x, y, _, _ = load()
    torch.set_num_threads(6)
    library = Path('build/libspan.so').resolve()
    records, fixture_checks = {}, {}
    with threadpool_limits(limits=6):
        rows = np.r_[*[np.flatnonzero(y == k)[:8] for k in range(3)]]
        for name, constructor in [('full', FullReference), ('span', ReferenceSpan)]:
            model = constructor(library)
            fixture_checks[name] = {}
            for label, a, b in [('mini', x[rows].copy(), y[rows].copy()), ('full', x, y)]:
                fixture_checks[name][label] = fixed(model, old, a, b)
            for bad in (np.nan, np.inf, -np.inf):
                a = x[rows].copy();a[0, 0] = bad
                try: model.fit(a, y[rows].copy())
                except RuntimeError as exc:
                    if 'nonfinite source matrix' not in str(exc): raise
                else: raise ValueError('bad input accepted')
        tick = time.perf_counter()
        sklearn, sklearn_optimizer = sklearn_fit(x, y)
        baseline = np.column_stack((sklearn.coef_, sklearn.intercept_))
        baseline_s = time.perf_counter()-tick
        loss_ref, _, p_ref = objective(x, y, baseline)
        for name, constructor in [('full', FullReference), ('span', ReferenceSpan)]:
            tick = time.perf_counter()
            theta, stats, calls = constructor(library).fit(x, y)
            elapsed = time.perf_counter()-tick
            loss, gradient, p = objective(x, y, theta)
            source = parity(p, p_ref)
            held = compare('P01', theta, baseline)
            passed = (loss <= loss_ref+1e-5 and gradient <= 1e-5
                      and source['max_probability_error'] <= 1e-3 and source['non_tied_changes'] == 0
                      and all(r['passed'] for r in held.values()))
            row = dict(passed=bool(passed), wall_s=elapsed, objective=loss, max_gradient=gradient,
                       stats=stats, source_parity=source, held_parity=held, evaluations=calls)
            records[name] = row
            np.savez(f'reference_{name}_fit.npz', theta=theta)
            Path('reference_progress.json').write_text(json.dumps(records, indent=2))
            print(name, json.dumps(row), flush=True)
    report = dict(passed=all(r['passed'] for r in records.values()), records=records,
                  fixture_checks=fixture_checks, sklearn_s=baseline_s, sklearn_optimizer=sklearn_optimizer, matrix_sha256=MATRIX_SHA,
                  library_sha256=sha(library), no_held_accuracy=True, node=os.uname().nodename)
    Path('reference_smoke_result.json').write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise ValueError('strict reference objective/held parity failed')


if __name__ == '__main__':
    main()
