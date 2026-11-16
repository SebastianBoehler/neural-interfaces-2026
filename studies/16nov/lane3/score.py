"""Validate all §292C folds, then score each seed independently; no model fitting."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))
ARMS = ('off', 'matched', 'split')
BLOB = 'a939ace9aa1e229f08391ad8bb2d197b507ae2c519a50addf087f0151b2df5c3'
MODEL_MANIFEST = '1ee038dcf899456855968e045a95e720c90236734aca664ea9f889ce779754bd'
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
PERM_SHA = '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'
PARAMS = {'off': 5022403, 'matched': 5048467, 'split': 5048467}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_names():
    return {f'{a}_s{s}_{p}.npz' for a in ARMS for s in range(2) for p in PEOPLE}


def matrix_paths(directory):
    found = {p.name for p in directory.glob('*.npz')}
    if found != expected_names():
        raise ValueError(f'wrong output matrix: missing={sorted(expected_names()-found)} extra={sorted(found-expected_names())}')
    return {name: directory / name for name in sorted(found)}


def validate_fold(path, arm, seed, held, subjects):
    names = sorted(set(subjects.tolist()))
    held_name = next((p for p in names if p.split('/')[-1] == held), None)
    if held_name is None or len(names) != 10:
        raise ValueError(f'{path}: invalid person universe')
    fit_candidates = [p for p in names if p != held_name]
    val = fit_candidates[seed]
    fit = set(fit_candidates) - {val}
    if len(fit) != 8 or held_name in fit or val in fit:
        raise ValueError(f'{path}: invalid inferred partition')
    rows = np.flatnonzero(subjects == held_name)
    before = sha(path)
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'index', 'metadata', 'val_log'} | {f'proba_{o}' for o in ORDERS}:
            raise ValueError(f'{path}: wrong output fields')
        m = json.loads(str(z['metadata'].item()))
        identity = (m.get('arm'), m.get('seed'), m.get('held'), m.get('val_person'))
        if identity != (arm, seed, held, val) or not np.array_equal(z['index'], rows):
            raise ValueError(f'{path}: fold identity, validation person or rows differ')
        log = z['val_log']
        if (log.shape != (25,) or not np.isfinite(log).all() or
                m.get('best_epoch') != int(np.argmax(log)) or m.get('checkpoint_sha256') != BLOB or
                m.get('trainable') != PARAMS[arm] or tuple(m.get('orders', ())) != ORDERS):
            raise ValueError(f'{path}: training or model provenance differs')
        arrays = {}
        for order in ORDERS:
            a = z[f'proba_{order}']
            if (a.shape != (len(rows), 3) or not np.isfinite(a).all() or
                    (a < 0).any() or not np.allclose(a.sum(1), 1, atol=1e-6)):
                raise ValueError(f'{path}: invalid probability output for {order}')
            arrays[order] = a.copy()
    if sha(path) != before:
        raise ValueError(f'{path}: output changed while reading')
    return arrays


def verify_sources(root, weights):
    source = root / 'source.sha256'
    if sha(source) != MODEL_MANIFEST:
        raise ValueError('model source manifest hash differs')
    for line in source.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if sha(root / name) != digest:
            raise ValueError(f'model source mismatch: {name}')
    for path, digest in ((root / 'data/proteus_windows.npz', DATA_SHA),
                         (root / 'runs/shuffle_perms.npz', PERM_SHA), (weights, BLOB)):
        if sha(path) != digest:
            raise ValueError(f'input mismatch: {path}')


def verify_orders(root, n):
    with np.load(root / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        for order in ORDERS:
            name = f'{order[:-1]}|{order[-1]}|all_loso'
            if not np.array_equal(np.sort(z[name]), np.arange(n)):
                raise ValueError(f'invalid pinned order: {order}')


def load_helpers(root, collector):
    sys.path[:0] = [str(root / 'experiments'), str(collector / 'studies/14nov/newmember')]
    import protocol
    from score_s_v66p import person_accuracy, summarize
    return protocol, person_accuracy, summarize


def preflight(root, collector, weights):
    verify_sources(root, weights)
    with np.load(root / 'data/proteus_windows.npz', allow_pickle=False) as z:
        subjects = z['subject'].copy()
    verify_orders(root, len(subjects))
    p = root / 'outputs/off_s0_P01.npz'
    values = validate_fold(p, 'off', 0, 'P01', subjects)
    _, person_accuracy, _ = load_helpers(root, collector)
    n = len(values['within0'])
    # Synthetic labels only: this verifies scorer identity without reading held labels or accuracy.
    synthetic = np.zeros(n, dtype=int)
    fake_people = np.full(n, 'P01')
    fake_cells = np.full(n, 'P01-cell')
    a = person_accuracy(values['within0'], synthetic, fake_cells, fake_people, ('P01',))
    b = person_accuracy(values['within0'], synthetic, fake_cells, fake_people, ('P01',))
    if not np.array_equal(a - b, np.zeros(1)):
        raise ValueError('off-minus-off metric identity failed')
    print(json.dumps({'preflight': 'OK', 'file_sha256': sha(p), 'rows': n,
                      'orders': len(ORDERS), 'metric_identity': 0, 'held_labels_read': False}))


def score(root, collector, weights):
    verify_sources(root, weights)
    paths = matrix_paths(root / 'outputs')
    protocol, person_accuracy, summarize = load_helpers(root, collector)
    with np.load(root / 'data/proteus_windows.npz', allow_pickle=False) as z:
        subjects = z['subject'].copy()
    n = len(subjects)
    verify_orders(root, n)
    merged = {(arm, seed, order): np.full((n, 3), np.nan) for arm in ARMS for seed in range(2) for order in ORDERS}
    hashes = {}
    for arm in ARMS:
        for seed in range(2):
            for held in PEOPLE:
                name = f'{arm}_s{seed}_{held}.npz'; path = paths[name]
                v = validate_fold(path, arm, seed, held, subjects)
                rows = np.flatnonzero(np.char.endswith(subjects, '/' + held))
                for order in ORDERS:
                    merged[arm, seed, order][rows] = v[order]
                hashes[name] = sha(path)
    if any(not np.isfinite(v).all() for v in merged.values()):
        raise ValueError('incomplete prediction rows')
    # Only now read true labels: the complete model matrix has passed all guards.
    d = protocol.load(); y = d['y']
    if not np.array_equal(subjects, d['subject']):
        raise ValueError('protocol subject rows differ')
    person = np.array([s.split('/')[-1] for s in subjects])
    cell = np.char.add(np.char.add(d['stream'], '|'), d['interface'])
    if any(set(y[cell == c].tolist()) != {0, 1, 2} for c in np.unique(cell)):
        raise ValueError('person helper differs from official metric when a cell lacks a class')
    mask = np.ones(n, dtype=bool)
    per_person, official = {}, {}
    for arm in ARMS:
        for order in ORDERS:
            guesses = [merged[arm, seed, order] for seed in range(2)]
            per_person[arm, order] = np.mean([person_accuracy(a, y, cell, person, PEOPLE) for a in guesses], axis=0)
            official[arm, order] = float(np.mean([100 * protocol.score(y, a.argmax(1), d, mask)['balanced_accuracy_cell_mean'] for a in guesses]))
    report = {'official_equal_cell_by_order': {a: {o: official[a, o] for o in ORDERS} for a in ARMS},
              'official_equal_cell_scheme_equal': {a: float(np.mean([
                  np.mean([official[a, o] for o in ORDERS if o.startswith(s)])
                  for s in ('within', 'full', 'runs')])) for a in ARMS},
              'absolute_people_order_intervals': {a: summarize(
                  np.stack([per_person[a, o] for o in ORDERS]), ORDERS, PEOPLE) for a in ARMS},
              'people_order_intervals': {}, 'output_sha256': hashes, 'model_manifest_sha256': MODEL_MANIFEST,
              'data_sha256': DATA_SHA, 'permutations_sha256': PERM_SHA, 'checkpoint_sha256': BLOB,
              'aggregation': 'per-seed argmax, per-seed metrics, then mean of two seeds',
              'limits': 'P04/P10 are held-out new-person folds, not returning sessions; training IDs are inferred from hash-verified partition code and recorded held/validation IDs, not stored individually in outputs.'}
    for a, b in (('split', 'matched'), ('split', 'off'), ('matched', 'off')):
        report['people_order_intervals'][f'{a}_minus_{b}'] = summarize(
            np.stack([per_person[a, o] - per_person[b, o] for o in ORDERS]), ORDERS, PEOPLE)
    return report


def main():
    root = Path(os.environ['LANE3_MODEL_ROOT']).resolve(strict=True)
    collector = Path(__file__).resolve().parents[3]
    weights = Path(os.environ['LANE3_WEIGHTS']).resolve(strict=True)
    if sys.argv[1:] == ['preflight']:
        return preflight(root, collector, weights)
    if sys.argv[1:] != ['score']:
        raise SystemExit('usage: score.py preflight | score')
    result = score(root, collector, weights)
    result['collector_sha256'] = sha(Path(__file__))
    result['collector_manifest_sha256'] = sha(collector / 'collector.sha256')
    target = collector / 'evidence/lane3_results.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'scored complete matrix -> {target}', flush=True)


if __name__ == '__main__':
    main()
