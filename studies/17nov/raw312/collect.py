"""Validate the entire registered producer matrix before opening held labels."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS'])
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'studies/14nov/newmember')]
from data import sha
from score_s_v66p import person_accuracy, summarize
from run import ARMS, ORDERS

PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))


def main():
    output = ROOT / 'output'
    expected = {f'{a}_{arm}_s{s}_{p}.npz' for a in ('fb', 'sh') for arm in ARMS for s in (0, 1) for p in PEOPLE}
    found = {p.name for p in output.glob('*.npz')}
    if found != expected:
        raise ValueError(f'incomplete matrix: missing={expected-found}, extra={found-expected}')
    with np.load(ROOT / 'data/proteus_windows.npz') as z:
        subjects = z['subject'].copy()
    n = len(subjects)
    merged = {(a, arm, s, o): np.full((n, 3), np.nan) for a in ('fb', 'sh') for arm in ARMS for s in (0, 1) for o in ORDERS}
    digests = {}
    for a in ('fb', 'sh'):
        for arm in ARMS:
            for s in (0, 1):
                for person in PEOPLE:
                    path = output / f'{a}_{arm}_s{s}_{person}.npz'
                    with np.load(path) as z:
                        meta = json.loads(str(z['metadata']))
                        rows = np.flatnonzero(np.char.endswith(subjects, '/' + person))
                        if not np.array_equal(rows, z['index']):
                            raise ValueError('held canonical rows differ')
                        target = str(subjects[rows[0]])
                        if target in meta['fit_people'] or target in meta['inner_fit_people']:
                            raise ValueError('outer source leakage')
                        if len(meta['fit_people']) != 9 or len(meta['inner_fit_people']) != 8:
                            raise ValueError('source split differs')
                        if (meta['held'], meta['architecture'], meta['arm'], meta['seed']) != (person, a, arm, s):
                            raise ValueError('arm identity differs')
                        if meta['source_manifest'] != sha(ROOT / 'source.sha256'):
                            raise ValueError('source manifest differs')
                        if sha(path.with_suffix('.pt')) != meta['checkpoint_sha256']:
                            raise ValueError('checkpoint changed')
                        for raw, expected_hash in meta['raw_hashes'].items():
                            if sha(raw) != expected_hash:
                                raise ValueError('raw source cache changed')
                        for o in ORDERS:
                            p = z[f'proba_{o}']
                            if p.shape != (len(rows), 3) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
                                raise ValueError('invalid held probabilities')
                            merged[a, arm, s, o][rows] = p
                    digests[path.name] = sha(path)
    if any(not np.isfinite(p).all() for p in merged.values()):
        raise ValueError('incomplete canonical predictions')
    # Existing owning evaluator/metric and nested uncertainty; labels opened only here.
    import protocol
    d = protocol.load(); y = d['y']
    person = np.array([s.split('/')[-1] for s in subjects])
    cell = np.char.add(np.char.add(d['stream'], '|'), d['interface'])
    mask = np.ones(n, dtype=bool)
    reports = {}
    for a in ('fb', 'sh'):
        official, pp = {}, {}
        for arm in ARMS:
            for o in ORDERS:
                guesses = [merged[a, arm, s, o] for s in (0, 1)]
                official[arm, o] = float(np.mean([100 * protocol.score(y, p.argmax(1), d, mask)['balanced_accuracy_cell_mean'] for p in guesses]))
                pp[arm, o] = np.mean([person_accuracy(p, y, cell, person, PEOPLE) for p in guesses], 0)
        reports[a] = dict(official_equal_cell_by_order={arm: {o: official[arm, o] for o in ORDERS} for arm in ARMS},
            effects={f'{arm}_minus_{base}': summarize(np.stack([pp[arm, o]-pp[base, o] for o in ORDERS]), ORDERS, PEOPLE)
                     for arm, base in (('T', 'C'), ('T', 'S'), ('T', 'F'), ('F', 'S'), ('C', 'S'))})
    report = dict(representations=reports, hashes=digests,
        scope='whole new-person alternatives;no returning or current-chain conclusion',
        aggregation='independent seed argmax/metric;then average seed metrics')
    with (ROOT / 'whole_results.json').open('x') as f:
        json.dump(report, f, indent=2)
    print('complete guarded whole-model matrix scored', flush=True)


if __name__ == '__main__':
    main()
