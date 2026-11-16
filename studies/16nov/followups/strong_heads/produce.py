"""One held person/seed: nested selection, source refit and eight causal streams."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ['S308_BASE']).resolve(strict=True)
CACHE = Path(os.environ['S308_CACHE']).resolve(strict=True)
OUT = Path(os.environ['S308_OUT']).resolve(strict=True)
sys.path[:0] = [str(HERE), str(BASE / 'experiments'), str(BASE / 'studies/16nov/followups/native_core')]
import protocol  # noqa: E402
from causal_native import CausalNative  # noqa: E402
from definitions import ARMS, ORDERS  # noqa: E402
from export import ridge_export, structured_export  # noqa: E402
from geometry import source_prior, sessions_in_order  # noqa: E402
from native import NativeHead  # noqa: E402
from split import source_partition  # noqa: E402
from train import fit_ridge_grid, fit_structured, refit_ridge, refit_structured  # noqa: E402

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stream_features(native, prior, shard, permutation, streams):
    rows = shard['rows']
    result = np.empty((len(rows), 44239), dtype=np.float64)
    filled = np.zeros(len(rows), dtype=bool)
    for session_rows in sessions_in_order(permutation, rows, streams):
        local = np.searchsorted(rows, session_rows)
        if not np.array_equal(rows[local], session_rows) or np.any(filled[local]):
            raise ValueError('session row mapping is invalid')
        result[local] = native.causal(prior, np.ascontiguousarray(shard['cov'][local]),
                                      np.ascontiguousarray(shard['slow'][local]))
        filled[local] = True
    if not filled.all() or not np.isfinite(result).all():
        raise ValueError('causal feature stream incomplete')
    return result


def main(task):
    if not 0 <= task < 20:
        raise ValueError('task must encode one of ten outer people and two seeds')
    person, seed = divmod(task, 2)
    label = f'P{person + 1:02d}_s{seed}'
    target = OUT / 'shards' / f'{label}.npz'
    checkpoint_dir = OUT / 'checkpoints' / label
    if target.exists() or checkpoint_dir.exists():
        raise FileExistsError(f'existing result for {label}')
    checkpoint_dir.mkdir(parents=True)
    d = protocol.load()
    people = np.array([int(s.rsplit('P', 1)[1]) - 1 for s in d['subject']])
    inner_train, inner_mask, outer_mask = source_partition(people, person, seed)
    inner_person = int(np.unique(people[inner_mask])[0])
    device = int(os.environ['S308_DEVICE'])
    native = CausalNative(device)
    torch_device = 'cuda' if device == 1 else 'cpu'
    priors, counts, parts, inputs, selected = [], [], [], {}, {}
    core_hash, train_hash = digest(Path(os.environ['NATIVE_CORE_LIB'])), digest(Path(os.environ['NATIVE_TRAIN_LIB']))
    for p in range(10):
        path = CACHE / f'P{p + 1:02d}.npz'
        inputs[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            rows = z['rows'].copy()
            if (meta['person'] != f'P{p + 1:02d}' or not np.array_equal(rows, np.flatnonzero(people == p)) or
                    meta['held_accuracy_computed'] is not False or
                    meta['rows_sha256'] != hashlib.sha256(rows.tobytes()).hexdigest() or
                    meta['native_core_sha256'] != core_hash or meta['native_train_sha256'] != train_hash):
                raise ValueError('cache metadata or person identity mismatch')
            priors.append(z['prior'].copy())
            counts.append(len(rows))
            if p in (person, inner_person):
                selected[p] = {'rows': rows, 'cov': z['cov'].copy(), 'slow': z['slow'].copy()}
            if p != person:
                parts.append((p, rows, z['features'].astype(np.float32)))
    priors, counts = np.asarray(priors), np.asarray(counts)
    if not np.allclose(source_prior(priors, counts, [person]),
                       native.combine(np.ascontiguousarray(priors), counts, person), atol=1e-12, rtol=1e-12):
        raise ValueError('population-prior formula differs from native reference')
    eight_prior = source_prior(priors, counts, [person, inner_person])
    nine_prior = source_prior(priors, counts, [person])
    x8 = np.concatenate([features for p, _, features in parts if p != inner_person])
    rows8 = np.concatenate([rows for p, rows, _ in parts if p != inner_person])
    x9 = np.concatenate([features for _, _, features in parts])
    rows9 = np.concatenate([rows for _, rows, _ in parts])
    if not np.array_equal(np.sort(rows8), np.flatnonzero(inner_train)) or not np.array_equal(np.sort(rows9), np.flatnonzero(~outer_mask)):
        raise ValueError('inner or final training rows violate held-person isolation')
    with np.load(BASE / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        permutations = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    validation = stream_features(native, eight_prior, selected[inner_person], permutations['within0'], d['stream'])
    y8, y9, yvalid = d['y'][rows8], d['y'][rows9], d['y'][selected[inner_person]['rows']]
    selections, models = {}, {}
    for arm in ARMS:
        start = time.monotonic()
        path = checkpoint_dir / f'{arm}.pt'
        if arm.startswith('S'):
            nonlinear = arm == 'Stanh'
            _, _, epoch, losses = fit_structured(x8, y8, validation, yvalid, nonlinear, seed, device=torch_device)
            model, scaler = refit_structured(x9, y9, nonlinear, seed, epoch, device=torch_device)
            structured_export(model, *scaler, validation[:1], path)
            selections[arm] = {'epoch': epoch, 'inner_logloss': losses}
        else:
            kind = {'Rlinear': 'linear', 'Rquadratic': 'quadratic', 'Rfourier': 'fourier'}[arm]
            chosen = fit_ridge_grid(x8, y8, validation, yvalid, kind, seed)
            model = refit_ridge(x9, y9, kind, seed, chosen['strength'], chosen['temperature'])
            ridge_export(model, validation[:1], path)
            selections[arm] = {'strength': chosen['strength'], 'temperature': chosen['temperature'],
                               'inner_logloss': chosen['validation']}
        models[arm] = {'path': str(path.relative_to(OUT)), 'sha256': digest(path), 'seconds': time.monotonic() - start}
        del model
    del x8, x9, validation
    reference = Path(os.environ['S308_REFERENCES']) / f'P{person + 1:02d}.npz'
    manifest = json.loads((reference.parent / 'manifest.json').read_text())
    if digest(reference) != manifest['files'][reference.name]:
        raise ValueError('reference head checksum mismatch')
    with np.load(reference, allow_pickle=False) as z:
        fixed = {key: np.ascontiguousarray(z[key]) for key in z.files}
    if not np.array_equal(fixed['fit_rows'], np.flatnonzero(~outer_mask)):
        raise ValueError('reference head did not use exact nine-person source fold')
    handles = {arm: NativeHead(os.environ['S308_LIB'], OUT / models[arm]['path'], device) for arm in ARMS}
    result = {'rows': selected[person]['rows']}
    try:
        for order in ORDERS:
            features = stream_features(native, nine_prior, selected[person], permutations[order], d['stream'])
            reference_p = native.predict(features, fixed['mean'], fixed['scale'], fixed['coef'], fixed['intercept'])
            for arm, model in handles.items():
                probs = model.predict(features)
                probs /= probs.sum(axis=1, keepdims=True)
                delta = probs - reference_p
                if not np.isfinite(delta).all() or not np.allclose(delta.sum(1), 0, atol=1e-10):
                    raise ValueError('invalid full-system member delta')
                result[f'{arm}|{order}'] = delta
    finally:
        for handle in handles.values():
            handle.close()
    metadata = {'outer_person': person, 'inner_person': inner_person, 'seed': seed, 'models': models,
                'selection': selections, 'cache_sha256': inputs, 'reference_sha256': digest(reference),
                'source_sha256': digest(Path(os.environ['S308_MANIFEST'])), 'core_sha256': core_hash,
                'head_library_sha256': digest(Path(os.environ['S308_LIB'])), 'device': device,
                'held_accuracy_computed': False}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        np.savez(f, **result, metadata=json.dumps(metadata, sort_keys=True))
    print(f'{label}: all registered head streams prepared; no outer score', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: produce.py TASK_INDEX')
    main(int(sys.argv[1]))
