"""Matched cached-feature full-fold training times to a common objective target."""
import json
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'experiments')]
os.environ['NI26_ROOT'] = str(ROOT)
import protocol  # noqa: E402
from bridge import Native  # noqa: E402


def main():
    target = Path(os.environ['S301_TRAIN_BENCH_OUTPUT'])
    if target.exists():
        raise FileExistsError(target)
    d = protocol.load()
    folds = [(tr, te) for held, tr, te in protocol.folds(d, 'all_loso') if held == 'P01']
    if len(folds) != 1:
        raise ValueError('P01 fold unavailable')
    tr, te = folds[0]
    rows, parts = [], []
    cache_seconds = 0.
    producer_lock = Path(os.environ['S301_BENCH_PRODUCER']) / 'producer.sha256'
    source_sha = hashlib.sha256(producer_lock.read_bytes()).hexdigest()
    for i in range(2, 11):
        with np.load(ROOT / f'runs/s301_native_inputs/P{i:02d}.npz', allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            if (meta['person'] != f'P{i:02d}' or meta['source_sha256'] != source_sha or
                    meta['held_accuracy_computed'] is not False):
                raise ValueError('cached person provenance differs')
            rows.append(z['rows'].copy()); parts.append(z['features'].copy())
            cache_seconds += float(meta['seconds'])
    index = np.concatenate(rows); position = np.argsort(index)
    if not np.array_equal(index[position], np.flatnonzero(tr)) or np.any(te[index]):
        raise ValueError('training folds differ or include the held person')
    raw = np.ascontiguousarray(np.concatenate(parts)[position])
    y = np.ascontiguousarray(d['y'][index[position]], dtype=np.int64)
    with np.load(Path(os.environ['S301_ALL_REFERENCES']) / 'P01.npz', allow_pickle=False) as z:
        ref = {k: np.ascontiguousarray(z[k]) for k in z.files}
    native = Native()
    native_x = native.standardize(raw, ref['mean'], ref['scale'])
    python_x = np.ascontiguousarray((raw - ref['mean']) / ref['scale'])
    np.testing.assert_allclose(native_x, python_x, atol=1e-12, rtol=1e-12)
    ref_theta = np.ascontiguousarray(np.column_stack((ref['coef'], ref['intercept'])))
    ref_loss, _ = native.objective(native_x, y, ref_theta)
    target_loss = ref_loss + 1e-5
    results = []
    for order in (('python', 'native'), ('native', 'python'), ('python', 'native')):
        for method in order:
            t0 = time.perf_counter()
            if method == 'python':
                model = LogisticRegression(C=.3, max_iter=2000, tol=1e-5).fit(python_x, y)
                theta = np.ascontiguousarray(np.column_stack((model.coef_, model.intercept_)))
                calls = int(model.n_iter_.max())
                count_kind = 'sklearn optimizer iterations'
            else:
                theta, _, _, calls = native.fit(native_x, y)
                count_kind = 'native objective evaluations'
            elapsed = time.perf_counter() - t0
            loss, grad = native.objective(native_x, y, theta)
            maximum = float(np.max(np.abs(grad)))
            if not np.isfinite([loss, maximum]).all() or loss > target_loss:
                raise ValueError(f'{method} did not reach common objective target: {loss=} {target_loss=}')
            results.append({'method': method, 'solve_wall_s': elapsed, 'final_objective': loss,
                            'objective_target': target_loss, 'max_gradient': maximum,
                            'count': calls, 'count_kind': count_kind,
                            'time_to_target_upper_bound_s': elapsed})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        json.dump({'paired_runs': results, 'training_person_cache_generation_s_sum': cache_seconds,
                   'reference_head_sha256': hashlib.sha256((Path(os.environ['S301_ALL_REFERENCES']) / 'P01.npz').read_bytes()).hexdigest(),
                   'producer_source_sha256': source_sha,
                   'cache_scope': 'nine native person preparations including covariances, priors, whole-session features',
                   'target_note': 'full solve time is an upper bound on first reaching the common target; neither solver exposes an identical accepted-iterate target callback',
                   'held_accuracy_computed': False}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'matched training timings saved: {target}; no held score', flush=True)


if __name__ == '__main__':
    main()
