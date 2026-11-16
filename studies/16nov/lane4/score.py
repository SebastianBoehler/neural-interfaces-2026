"""§292D input validation and scoring through the existing §292B/§281 helpers."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('lane2_score', ROOT / 'studies/16nov/lane2/score.py')
S = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(S)
ORDERS = tuple([f'within{i}' for i in range(4)] + [f'{s}{i}' for s in ('full', 'runs') for i in range(2)])
PEOPLE, BASES = S.PEOPLE, S.BASES
RUNS = ROOT / 'runs/lane4'


def sha(path):
    return S.sha(path)


def read_manifest(path):
    items = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name in items:
            raise ValueError(f'duplicate manifest path: {name}')
        items[name] = digest
    return items


def load_outputs(d):
    n = len(d['y']); modes = ('frozen', 'diag', 'full')
    files = {f'fold_s{s}_{p}.npz' for s in range(3) for p in PEOPLE}
    found = {p.name for p in RUNS.glob('fold_*.npz')}
    if files != found:
        raise ValueError(f'output matrix incomplete: missing {sorted(files-found)}, extra {sorted(found-files)}')
    remote = read_manifest(RUNS / 'outputs.sha256')
    if set(remote) != {f'runs/lane4/{x}' for x in files}:
        raise ValueError('remote output manifest is not the exact fold matrix')
    inputs = read_manifest(RUNS / 'inputs.sha256')
    for name in ('data/proteus_windows.npz', 'runs/shuffle_perms.npz', 'experiments/protocol.py',
                 'experiments/deep.py', 'experiments/session_bn.py', 'studies/15nov/tsm288/spd.py',
                 'studies/16nov/lane4/run.py', 'studies/16nov/lane4/adapt.py'):
        if name not in inputs or sha(ROOT / name) != inputs[name]:
            raise ValueError(f'input/source hash differs: {name}')
    perms = np.load(ROOT / 'runs/shuffle_perms.npz')
    merged = {(seed, mode, order): np.full((n, 3), np.nan) for seed in range(3) for mode in modes for order in ORDERS}
    stats = []; max_early_diff = 0.0
    folds = {p: (tr, te) for p, tr, te in S.protocol.folds(d, 'all_loso')}
    for seed in range(3):
        for person in PEOPLE:
            tr, te = folds[person]
            name = f'fold_s{seed}_{person}.npz'; path = RUNS / name
            if sha(path) != remote[f'runs/lane4/{name}']:
                raise ValueError(f'{name}: transferred hash differs')
            meta_path = path.with_suffix('.json')
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'].item()))
                if meta != json.loads(meta_path.read_text()):
                    raise ValueError(f'{name}: sidecar/NPZ mismatch')
                if meta['held'] != person or meta['seed'] != seed:
                    raise ValueError(f'{name}: fold identity mismatch')
                if set(meta['fit_people']) != set(d['subject'][tr]) or set(meta['fit_people']) & set(d['subject'][te]):
                    raise ValueError(f'{name}: training people mismatch')
                for order in ORDERS:
                    scheme = order.rstrip('0123456789'); number = order[len(scheme):]
                    perm = perms[f'{scheme}|{number}|all_loso'].astype(int)
                    if not np.array_equal(np.sort(perm), np.arange(n)):
                        raise ValueError(f'{order}: invalid permutation')
                    if meta['order_sha'][order] != hashlib.sha256(perm.tobytes()).hexdigest():
                        raise ValueError(f'{name}: pinned order hash mismatch')
                    rows = perm[te[perm]]; arrays = {}
                    for mode in modes:
                        if not np.array_equal(z[f'{order}|{mode}|index'], rows):
                            raise ValueError(f'{name}: order index mismatch {order}/{mode}')
                        a = z[f'{order}|{mode}|proba']
                        if a.shape != (len(rows), 3) or not np.isfinite(a).all() or not np.allclose(a.sum(1), 1, atol=1e-6):
                            raise ValueError(f'{name}: invalid probability {order}/{mode}')
                        arrays[mode] = a; merged[seed, mode, order][rows] = a
                    for sess in dict.fromkeys(d['stream'][rows]):
                        loc = np.flatnonzero(d['stream'][rows] == sess)
                        for mode in ('diag', 'full'):
                            diff = float(np.max(np.abs(arrays[mode][loc[:32]] - arrays['frozen'][loc[:32]])))
                            max_early_diff = max(max_early_diff, diff)
                            if diff > 1e-7:
                                raise ValueError(f'{name}: changed pre-update {order}/{mode}/{sess}: {diff}')
                    for mode in modes:
                        logs = meta[f'{order}|{mode}']
                        expected_sessions = set(map(str, d['stream'][rows]))
                        if {entry['session'] for entry in logs} != expected_sessions or len(logs) != len(expected_sessions):
                            raise ValueError(f'{name}: session logs incomplete')
                        for entry in logs:
                            sess = entry['session']; count = int(np.sum(d['stream'][rows] == sess))
                            expected_updates = 3 * len(range(32, count, 8)) if mode != 'frozen' else 0
                            if entry['updates'] != expected_updates or entry['bias_norm'] > 0.500001:
                                raise ValueError(f'{name}: update count/bound mismatch')
                            if expected_updates and (entry['grad_max'] <= 0 or entry['bias_norm'] <= 0):
                                raise ValueError(f'{name}: adaptation did not change')
                            stats.append((person, seed, order, mode, sess, count, entry['seconds']))
    if any(not np.isfinite(a).all() for a in merged.values()):
        raise ValueError('missing output rows')
    return merged, stats, max_early_diff


def main():
    d = S.protocol.load(); y = d['y']; n = len(y)
    candidate, timings, max_early_diff = load_outputs(d)
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        if not np.array_equal(z['y|all_loso'], y):
            raise ValueError('replay and official labels differ')
        cell, person = z['cell|all_loso'], z['person|all_loso']
    modes = ['frozen', 'diag', 'full']; reference_error = None
    try:
        S.ORDERS, S.REF = ORDERS, ROOT / 'runs/lane4_reference'
        ref = S.check_reference(n); modes.append('reference')
    except (OSError, ValueError, KeyError) as exc:
        ref, reference_error = None, str(exc)
    allrows = np.ones(n, dtype=bool); pp, official = {}, {}
    for order in ORDERS:
        for mode in modes:
            preds = [ref[order, b] for b in BASES] if mode == 'reference' else [candidate[s, mode, order] for s in range(3)]
            pp[mode, order] = np.mean([S.person_accuracy(x, y, cell, person, PEOPLE) for x in preds], axis=0)
            official[mode, order] = np.mean([100 * S.protocol.score(y, x.argmax(1), d, allrows)['balanced_accuracy_cell_mean'] for x in preds])
    def difference(a, b):
        return S.summarize(np.stack([pp[a, o] - pp[b, o] for o in ORDERS]), ORDERS, PEOPLE)
    def scheme_equal(values):
        return float(np.mean([np.mean([values[o] for o in ORDERS if o.startswith(s)]) for s in ('within', 'full', 'runs')]))
    points = {m: {o: official[m, o] for o in ORDERS} for m in modes}
    result = {'preupdate_max_probability_difference': max_early_diff,
              'official_equal_cell_points_by_order': points,
              'official_equal_cell_scheme_equal': {m: scheme_equal(points[m]) for m in modes},
              'diag_minus_frozen_people_order': difference('diag', 'frozen'),
              'full_minus_frozen_people_order': difference('full', 'frozen'),
              'full_minus_diag_people_order': difference('full', 'diag'),
              'seed_aggregation': 'mean of three separately argmax-scored classifiers',
              'interval': '80% person/order bootstrap; training seeds averaged first',
              'reference_error': reference_error,
              'reference_limit': 'unrecognized-person route emulation with full personalization; not returning route',
              'stream_timing_note': 'mean of per-session ms/window; stream readout and updates only, excludes feature extraction',
              'stream_timing_ms_per_window': {m: float(np.mean([1000 * t[6] / t[5] for t in timings if t[3] == m]))
                                              for m in ('frozen', 'diag', 'full')}}
    if ref is not None:
        result['frozen_minus_reference_people_order'] = difference('frozen', 'reference')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
