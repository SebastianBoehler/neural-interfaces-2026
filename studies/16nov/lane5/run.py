"""Native whole-system LOSO runner: outputs probabilities, never scores."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get('NI26_WS', Path(__file__).resolve().parents[3])).resolve()
sys.path[:0] = [str(ROOT / 'studies/16nov/lane5'), str(ROOT / 'experiments')]
import protocol  # noqa: E402
from bridge import Native, Session  # noqa: E402

ARMS = (('temporal_ce', 1, 0), ('connectivity_ce', 2, 0),
        ('hybrid_ce', 3, 0), ('hybrid_consistency', 3, 1))
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def task_config(task):
    if not 0 <= task < 80:
        raise ValueError('task must be in 0..79')
    return ARMS[task // 20], (task % 20) // 10, task % 10


def probability(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def validate_fit_status(status, architecture, consistency):
    s = status
    if (s['steps'] <= 0 or not np.isfinite([s['consistency_sum'], s['projection_grad_max'], s['consistency_grad_max']]).all() or
            (consistency and (s['consistency_sum'] <= 0 or s['consistency_grad_max'] <= 0)) or
            (not consistency and (s['consistency_sum'] != 0 or s['consistency_grad_max'] != 0)) or
            (architecture in (2, 3) and s['projection_grad_max'] <= 0)):
        raise ValueError(f'inactive native training mechanism: {s}')


def stream(population, x):
    """Independent handles, one current raw window per native call."""
    started = time.monotonic()
    static, adapt = Session(population), Session(population)
    initialization = time.monotonic() - started
    a, b = np.empty((len(x), 3)), np.empty((len(x), 3))
    t0 = time.monotonic()
    static_s = adaptive_s = 0.0
    for t, window in enumerate(x):
        started = time.monotonic()
        a[t] = probability(static.step(window, False))
        dt_static = time.monotonic() - started
        started = time.monotonic()
        b[t] = probability(adapt.step(window, True))
        if t >= 5:
            static_s += dt_static
            adaptive_s += time.monotonic() - started
    elapsed = time.monotonic() - t0
    if not np.allclose(a[:40], b[:40], atol=1e-6, rtol=1e-6):
        raise ValueError('adaptation changed predictions before the first update')
    sa, sb = static.solver.status(), adapt.solver.status()
    expected = len(range(40, len(x), 20))
    if (sa['windows'] != len(x) or sa['updates'] != 0 or sb['windows'] != len(x) or
            sb['updates'] != expected or not np.isfinite([sb['max_grad'], sb['delta']]).all() or
            sb['delta'] > .100001 or (expected and (sb['max_grad'] <= 0 or sb['delta'] <= 0))):
        raise ValueError(f'invalid native session status: static={sa}, adaptive={sb}')
    if expected and sb['prefix_count'] != min(64, 40 + 20 * (expected - 1)):
        raise ValueError(f'wrong adaptation prefix size: {sb}')
    static.solver.close(); adapt.solver.close()
    return a, b, {'static': sa, 'adaptive': sb, 'seconds': elapsed,
                  'session_initialization_s': initialization, 'warm_windows': max(0, len(x) - 5),
                  'warm_static_s': static_s, 'warm_adaptive_s': adaptive_s,
                  'timing_scope': 'paired Python/native calls including preprocessing and transfers; excludes first five windows'}


def orders(root, n):
    with np.load(root / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        result = {}
        for o in ORDERS:
            p = z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64)
            if not np.array_equal(np.sort(p), np.arange(n)):
                raise ValueError(f'invalid permutation {o}')
            result[o] = p
        return result


def main(task):
    (arm, architecture, consistency), seed, index = task_config(task)
    library = Path(os.environ['LANE5_LIB']).resolve(strict=True)
    manifest = Path(os.environ['LANE5_MANIFEST']).resolve(strict=True)
    device = int(os.environ['LANE5_DEVICE'])
    if device not in (0, 1):
        raise ValueError('LANE5_DEVICE must be 0 or 1')
    d = protocol.load(); folds = list(protocol.folds(d, 'all_loso'))
    if len(folds) != 10:
        raise ValueError('expected ten held people')
    held, train, test = folds[index]
    subjects = d['subject']; fit_people = sorted(set(subjects[train]))
    if len(fit_people) != 9 or set(fit_people) & set(subjects[test]):
        raise ValueError('training and held people are not disjoint')
    person_ids = np.array([int(s.split('/')[-1][1:]) for s in subjects[train]], dtype=np.int32)
    held_id = int(held[1:])
    permutations = orders(ROOT, len(subjects))
    output = ROOT / f'runs/lane5/{arm}_s{seed}_{held}.npz'
    checkpoint = ROOT / f'runs/lane5_checkpoints/{arm}_s{seed}_{held}.pt'
    if output.exists() or checkpoint.exists():
        raise FileExistsError(f'output or checkpoint exists: {output}, {checkpoint}')
    if device == 1 and not torch.cuda.is_available():
        raise RuntimeError('native CUDA requested but PyTorch reports no GPU')
    native = Native(library); solver = native.create(architecture, device, seed)
    t0 = time.monotonic()
    train_started = time.monotonic()
    solver.fit(d['X'][train], d['y'][train], person_ids, held_id, 60, consistency)
    training_seconds = time.monotonic() - train_started
    fit_status = solver.fit_status()
    validate_fit_status(fit_status, architecture, consistency)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    solver.save(checkpoint)
    checkpoint_sha = sha(checkpoint)
    params_sha = hashlib.sha256(solver.params().tobytes()).hexdigest()
    result, logs, order_hash = {}, {}, {}
    for order, perm in permutations.items():
        rows = perm[test[perm]]
        order_hash[order] = hashlib.sha256(perm.tobytes()).hexdigest()
        a, b = np.empty((len(rows), 3)), np.empty((len(rows), 3))
        entries = []
        for session in dict.fromkeys(d['stream'][rows]):
            loc = np.flatnonzero(d['stream'][rows] == session)
            a[loc], b[loc], status = stream(solver, d['X'][rows[loc]])
            entries.append({'session': str(session), **status})
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f'non-finite probabilities in {order}')
        result[f'{order}|index'] = rows
        result[f'{order}|static'], result[f'{order}|adaptive'] = a, b
        logs[order] = entries
    metadata = {'arm': arm, 'architecture': architecture, 'consistency': consistency, 'seed': seed,
                'held': held, 'fit_people': fit_people, 'orders': list(ORDERS), 'order_sha': order_hash,
                'status': logs, 'fit_status': fit_status, 'epochs': 60, 'device': device, 'elapsed_s': time.monotonic() - t0,
                'training_seconds': training_seconds, 'torch_version': str(torch.__version__),
                'torch_cuda': torch.version.cuda, 'gpu_name': torch.cuda.get_device_name(0) if device else None,
                'checkpoint_path': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': checkpoint_sha,
                'parameter_sha256': params_sha,
                'library_sha256': sha(library), 'manifest_sha256': sha(manifest),
                'data_sha256': sha(ROOT / 'data/proteus_windows.npz'),
                'permutations_sha256': sha(ROOT / 'runs/shuffle_perms.npz')}
    result['metadata'] = json.dumps(metadata, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f, **result)
    solver.close()
    print(f'saved {output}; no metrics computed', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: run.py TASK')
    main(int(sys.argv[1]))
