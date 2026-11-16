"""Score complete §292B outputs using existing official and §281 score helpers.

No training or inference is performed here. Refuse partial, mismatched or unverified
inputs; do not average probability vectors before computing a window's label.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'studies/14nov/newmember')]
import protocol  # noqa: E402
from score_s_v66p import BASES, person_accuracy, summarize  # noqa: E402

ORDERS = tuple(f'{name}{i}' for name in ('within', 'full', 'runs') for i in (0, 1))
PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))
RUNS = ROOT / 'runs/lane2'
REF = ROOT / 'runs/lane2_reference'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_outputs(d):
    n = len(d['y']); seeds = 3
    expected = {f'fold_{p}_s{s}.npz' for p in PEOPLE for s in range(seeds)}
    found = {p.name for p in RUNS.glob('fold_*.npz')}
    if found != expected:
        raise ValueError(f'wrong output matrix: missing={sorted(expected-found)} extra={sorted(found-expected)}')
    manifest = {}
    for line in (ROOT / 'studies/16nov/lane2/inputs.sha256').read_text().splitlines():
        digest, path = line.split(maxsplit=1); manifest[path] = digest
    included = ('data/proteus_windows.npz', 'runs/shuffle_perms.npz', 'experiments/protocol.py',
                'studies/16nov/lane2/model.py', 'studies/16nov/lane2/run.py', 'studies/16nov/lane2/stream.py')
    for name in included:
        if sha(ROOT / name) != manifest[name]:
            raise ValueError(f'local input hash differs: {name}')
    perms = np.load(ROOT / 'runs/shuffle_perms.npz')
    merged = {(s, a, o): np.full((n, 3), np.nan, np.float32)
              for s in range(seeds) for a in ('static', 'adaptive') for o in ORDERS}
    diagnostics = []
    for p in PEOPLE:
        rows = np.flatnonzero(np.char.endswith(d['subject'].astype(str), '/' + p))
        for seed in range(seeds):
            path = RUNS / f'fold_{p}_s{seed}.npz'
            with np.load(path, allow_pickle=False) as z:
                if str(z['person'].item()) != p or int(z['seed'].item()) != seed or not np.array_equal(z['index'], rows):
                    raise ValueError(f'{path}: fold identity/indices differ')
                recorded = json.loads(str(z['manifest'].item()))
                if any(recorded.get(k) != manifest[k] for k in included):
                    raise ValueError(f'{path}: source or input hash differs')
                diag = json.loads(str(z['diagnostics'].item()))
                if set(diag) != set(ORDERS):
                    raise ValueError(f'{path}: incomplete diagnostics')
                for order in ORDERS:
                    a, b, c = (z[f'{key}|{order}'] for key in ('static', 'off', 'adaptive'))
                    if any(x.shape != (len(rows), 3) or not np.isfinite(x).all() or
                           not np.allclose(x.sum(1), 1, atol=1e-6) for x in (a, b, c)):
                        raise ValueError(f'{path}: invalid {order} scores')
                    if not np.array_equal(a, b):
                        raise ValueError(f'{path}: update-off mismatch')
                    scheme = order.rstrip('0123456789'); num = order[len(scheme):]
                    perm = perms[f'{scheme}|{num}|all_loso'].astype(int)
                    if not np.array_equal(np.sort(perm), np.arange(n)):
                        raise ValueError(f'{order}: invalid permutation')
                    rank = np.empty(n, dtype=int); rank[perm] = np.arange(n)
                    for session in set(d['stream'][rows]):
                        sr = rows[d['stream'][rows] == session]
                        sr = sr[np.argsort(rank[sr])]
                        early = np.searchsorted(rows, sr[:20])
                        if not np.array_equal(a[early], c[early]):
                            raise ValueError(f'{path}: early-window update')
                    for sess in diag[order]['sessions']:
                        count = sum(sess['selected']); delta = sess['delta']
                        if delta > 0.100001 or (count > 0 and delta <= 0):
                            raise ValueError(f'{path}: inactive/uncapped eligible update')
                        diagnostics.append((p, seed, order, sess['session'], count, delta))
                    merged[seed, 'static', order][rows] = a
                    merged[seed, 'adaptive', order][rows] = c
    if any(not np.isfinite(v).all() for v in merged.values()):
        raise ValueError('unfilled rows')
    return merged, diagnostics


def check_reference(n):
    validated = json.loads((REF / 'reference_sha256.json').read_text())['outputs']
    output = {}
    for order in ORDERS:
        for base in BASES:
            name = f'control_{order}_{base}.npz'; key = f'runs/s281_v66_cp/{name}'
            p = REF / name
            if key not in validated or sha(p) != validated[key]:
                raise ValueError(f'{p}: reference provenance/hash mismatch')
            with np.load(p, allow_pickle=False) as z:
                if (str(z['arm']), str(z['order']), str(z['base'])) != ('control', order, base):
                    raise ValueError(f'{p}: reference identity mismatch')
                a = z['scores']
                if a.shape != (n, 3) or not np.isfinite(a).all():
                    raise ValueError(f'{p}: invalid reference scores')
                output[order, base] = a.copy()
    return output


def main():
    d = protocol.load(); y = d['y']; n = len(y)
    cand, diagnostics = check_outputs(d); reference = check_reference(n)
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        if not np.array_equal(z['y|all_loso'], y):
            raise ValueError('replay labels differ from official protocol')
        cell, person = z['cell|all_loso'], z['person|all_loso']
    allrows = np.ones(n, dtype=bool)
    pp, official = {}, {}
    for order in ORDERS:
        for arm in ('static', 'adaptive', 'reference'):
            predictions = [reference[order, b] for b in BASES] if arm == 'reference' else [cand[s, arm, order] for s in range(3)]
            pp[arm, order] = np.mean([person_accuracy(x, y, cell, person, PEOPLE) for x in predictions], axis=0)
            official[arm, order] = np.mean([100 * protocol.score(y, x.argmax(1), d, allrows)['balanced_accuracy_cell_mean']
                                             for x in predictions])
    def report_difference(a, b):
        return summarize(np.stack([pp[a, o] - pp[b, o] for o in ORDERS]), ORDERS, PEOPLE)
    summary = {'official_equal_cell_point_by_order': {a: {o: official[a, o] for o in ORDERS}
                                                    for a in ('static', 'adaptive', 'reference')},
               'adaptive_minus_static_people_order': report_difference('adaptive', 'static'),
               'static_minus_reference_people_order': report_difference('static', 'reference'),
               'adaptive_minus_reference_people_order': report_difference('adaptive', 'reference'),
               'seed_aggregation': 'mean of three separately argmax-scored classifiers',
               'reference_aggregation': 'mean of six separately argmax-scored base draws',
               'reference_limit': 'unrecognized-person route emulation with full personalisation; not real returning route',
               'no_update_sessions': [x for x in diagnostics if x[4] == 0],
               'diagnostic_session_orders': len(diagnostics)}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
