"""One person: fit all six source-only models and create eight causal streams."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ['S313_BASE']).resolve(strict=True)
SUPPORT = Path(os.environ['S313_SUPPORT']).resolve(strict=True)
KERNEL = Path(os.environ['S313_KERNEL']).resolve(strict=True)
sys.path[:0] = [str(HERE), str(BASE / 'experiments'), str(BASE / 'studies/16nov/followups/native_core'),
                str(SUPPORT / 'studies/16nov/followups/strong_heads'),
                str(KERNEL / 'studies/16nov/followups/head_kernel')]
import protocol  # noqa: E402
from causal_native import CausalNative  # noqa: E402
from design import ARMS, ORDERS, compose  # noqa: E402
from geometry import sessions_in_order  # noqa: E402
from input_data import digest, load_spectral, load_strong  # noqa: E402
from kernel import Kernel  # noqa: E402
from solver import Solver  # noqa: E402


def causal(native, prior, shard, permutation, streams):
    rows = shard['rows']
    result = np.empty((len(rows), 44239), dtype=np.float64)
    filled = np.zeros(len(rows), dtype=bool)
    for sr in sessions_in_order(permutation, rows, streams):
        local = np.searchsorted(rows, sr)
        if not np.array_equal(rows[local], sr) or np.any(filled[local]):
            raise ValueError('invalid causal session mapping')
        result[local] = native.causal(prior, np.ascontiguousarray(shard['cov'][local]),
                                      np.ascontiguousarray(shard['slow'][local]))
        filled[local] = True
    if not filled.all() or not np.isfinite(result).all():
        raise ValueError('incomplete strong causal stream')
    return result


def probability(features, model):
    logits = ((features - model['mean']) / model['scale']) @ model['theta'][:, :-1].T + model['theta'][:, -1]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all():
        raise ValueError('nonfinite model probability')
    return p


def main(person):
    if person not in range(10):
        raise ValueError('outer person must be 0–9')
    name = f'P{person+1:02d}'
    out = Path(os.environ['S313_OUT']).resolve(strict=True)
    target = out / 'folds' / f'{name}.npz'
    model_dir = out / 'models' / name
    if target.exists() or model_dir.exists():
        raise FileExistsError(f'{name}: output already exists')
    model_dir.mkdir(parents=True)
    d = protocol.load()
    x, train_rows, held, priors, counts, strong_hashes = load_strong(d, person)
    features, spectral_manifest = load_spectral()
    native, solver, checker = CausalNative(0), Solver(os.environ['S313_SOLVER_LIB']), Kernel(os.environ['S313_CHECK_LIB'])
    prior = native.combine(np.ascontiguousarray(priors), np.asarray(counts, dtype=np.int64), person)
    with np.load(BASE / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    rows = held['rows']
    if set(d['subject'][train_rows]) & set(d['subject'][rows]) or len(np.unique(d['subject'][train_rows])) != 9:
        raise ValueError('held-person isolation failed')
    source_y = np.ascontiguousarray(d['y'][train_rows], dtype=np.int64)
    source = {name: np.asarray(array[train_rows]) for name, array in features.items()}
    models, measurements = {}, {}
    for arm in ARMS:
        started = time.monotonic()
        matrix = compose(x, source['power'], source['real'], source['imag'], arm)
        scaler = StandardScaler().fit(matrix)
        standardized = np.ascontiguousarray(scaler.transform(matrix))
        theta, loss, gradmax, calls = solver.fit(standardized, source_y)
        independent_loss, independent_grad = checker.objective(standardized, source_y, theta, analytical=True)
        independent_max = float(np.max(np.abs(independent_grad)))
        if abs(loss-independent_loss) > 1e-9 or independent_max > 1e-5:
            raise ValueError(f'{name}/{arm}: objective/gradient independent check failed')
        model = {'mean': scaler.mean_, 'scale': scaler.scale_, 'theta': theta}
        model_path = model_dir / f'{arm}.npz'
        with model_path.open('xb') as f:
            np.savez(f, **model)
        models[arm] = model
        measurements[arm] = {'model_path': str(model_path.relative_to(out)), 'model_sha256': digest(model_path),
                             'features': matrix.shape[1], 'loss': loss, 'max_gradient': gradmax,
                             'independent_loss': independent_loss, 'independent_max_gradient': independent_max,
                             'objective_evaluations': calls, 'fit_seconds': time.monotonic()-started}
        del matrix, standardized, independent_grad
        print(f'{name}: source-only {arm} fitted; no held score', flush=True)
    del x, source
    references = Path(os.environ['S313_REFERENCES']).resolve(strict=True)
    ref_path = references / f'{name}.npz'
    ref_manifest = json.loads((references / 'manifest.json').read_text())
    if digest(ref_path) != ref_manifest['files'][ref_path.name]:
        raise ValueError('original rebuilt strong reference differs')
    with np.load(ref_path, allow_pickle=False) as z:
        if not np.array_equal(z['fit_rows'], train_rows):
            raise ValueError('reference head source fold differs')
        fixed = {k: np.ascontiguousarray(z[k]) for k in ('mean','scale','coef','intercept')}
    first_session = sessions_in_order(perms['within0'], rows, d['stream'])[0][:25]
    local = np.searchsorted(rows, first_session)
    if len(local) < 3:
        raise ValueError('real prefix smoke requires at least three windows')
    cov = np.ascontiguousarray(held['cov'][local]); slow = np.ascontiguousarray(held['slow'][local])
    first = native.causal(prior, cov, slow)
    changed = cov.copy(); changed[-1] = cov[0]
    altered = native.causal(prior, changed, slow)
    repeated = native.causal(prior, cov, slow)
    prefix_error = float(np.max(np.abs(first[:-1]-altered[:-1])))
    reset_error = float(np.max(np.abs(first-repeated)))
    if prefix_error > 1e-10 or reset_error > 1e-10:
        raise ValueError('strong feature prefix/reset smoke failed')
    output, spectral_reference = {'rows': rows}, None
    target_views = {key: np.asarray(value[rows]) for key, value in features.items()}
    for order in ORDERS:
        strong = causal(native, prior, held, perms[order], d['stream'])
        ref = native.predict(strong, fixed['mean'], fixed['scale'], fixed['coef'], fixed['intercept'])
        for arm in ARMS:
            matrix = compose(strong, target_views['power'], target_views['real'], target_views['imag'], arm)
            p = probability(matrix, models[arm])
            if arm == 'SPECTRAL':
                if spectral_reference is not None and not np.allclose(p, spectral_reference, atol=1e-12, rtol=0):
                    raise ValueError('spectral-only output changed with delivery order')
                spectral_reference = p.copy()
            output[f'prob|{arm}|{order}'] = p
            if arm != 'SPECTRAL':
                delta = p-ref
                if not np.allclose(delta.sum(1), 0, atol=1e-10, rtol=0):
                    raise ValueError('replacement posterior delta is not normalized')
                output[f'delta|{arm}|{order}'] = delta
            del matrix
        print(f'{name}/{order}: full causal streams generated; no held score', flush=True)
    metadata = {'person': name, 'device': 'CPU 0', 'source_rows_sha256': hashlib.sha256(train_rows.tobytes()).hexdigest(),
                'strong_cache_sha256': strong_hashes, 'spectral_manifest_sha256': spectral_manifest,
                'source_sha256': digest(Path(os.environ['S313_MANIFEST'])), 'reference_sha256': digest(ref_path),
                'solver_sha256': digest(Path(os.environ['S313_SOLVER_LIB'])), 'measurements': measurements,
                'prefix_max_abs': prefix_error, 'reset_max_abs': reset_error, 'held_accuracy_computed': False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        np.savez_compressed(f, **output, metadata=json.dumps(metadata, sort_keys=True))
    print(f'{name}: all 6 whole-model streams and 5 component deltas completed; no score', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: produce.py OUTER_PERSON_INDEX')
    main(int(sys.argv[1]))
