"""Validate complete frozen banks, apply source sanity, then open external accuracy."""
import argparse
import gzip
import csv
import json
from pathlib import Path

import numpy as np

from data import sha


def ba(y, p):
    assert set(y) == {0, 1}
    return 100 * np.mean([np.mean(p[y == c] == c) for c in (0, 1)])


def run(root, out):
    proof = json.loads((out / 'before_labels.json').read_text())
    assert proof['target_labels_opened'] is False and proof['prefix_reset_interleaving_and_immutable_guards']
    assert len(proof['source_exclusions']) == 10
    for record in proof['source_exclusions']:
        assert len(record['training_people']) == 9 and record['person'] not in record['training_people']
    for path, digest in proof['inputs'].items():
        assert sha(root / path) == digest
    for path, digest in proof['outputs'].items():
        assert sha(out / path) == digest
    src = np.load(root / 'data/proteus_windows.npz')
    bank = np.load(out / 'source_predictions.npz')
    people = sorted(set(src['subject']))
    keys = [f'{arm}|{order}' for arm in ('power', 'cov') for order in proof['orders']]
    assert set(bank.files) == set(keys)
    scores = np.empty((2, 8, 10))
    for a, arm in enumerate(('power', 'cov')):
        for o, order in enumerate(proof['orders']):
            p = bank[f'{arm}|{order}']
            assert p.shape == (2655, 2) and np.isfinite(p).all() and np.allclose(p.sum(1), 1)
            for j, person in enumerate(people):
                rows = np.flatnonzero((src['subject'] == person) & np.isin(src['y'], (1, 2)))
                cells = np.char.add(np.char.add(src['session'][rows], '|'), src['interface'][rows])
                scores[a, o, j] = np.mean([ba(src['y'][rows[cells == c]] - 1,
                                             p[rows[cells == c]].argmax(1)) for c in set(cells)])
    source = {'people': people, 'arms': ['power', 'cov'], 'orders': proof['orders'],
              'person_mean_ba': scores.mean(1).tolist(), 'mean_ba': scores.mean((1, 2)).tolist(),
              'source_cov_people_above50': int((scores[1].mean(0) > 50).sum())}
    (out / 'source_sanity.json').write_text(json.dumps(source, indent=2) + '\n')
    if source['mean_ba'][1] < 60 or source['source_cov_people_above50'] < 7:
        (out / 'blocked.json').write_text(json.dumps({'reason': 'registered source sanity failed', 'source': source}, indent=2))
        raise RuntimeError('Source sanity failed; external gold remains unopened by collector')
    target = np.load(root / 'data/scherer2015/windows.npz')
    foreign = np.load(out / 'target_predictions.npz')
    assert set(foreign.files) == set(keys)
    with gzip.open(root / 'runs/diag/external_semantics351/scherer_rows.tsv.gz', 'rt') as f:
        manifest = list(csv.DictReader(f, delimiter='\t'))
    assert np.array_equal(target['y'], [int(r['export_class']) for r in manifest])
    people_t = sorted(set(target['subject']))
    values = np.empty((2, 8, 9, 2))
    for a, arm in enumerate(('power', 'cov')):
        for o, order in enumerate(proof['orders']):
            p = foreign[f'{arm}|{order}']
            assert p.shape == (2130, 2) and np.isfinite(p).all() and np.allclose(p.sum(1), 1)
            for j, person in enumerate(people_t):
                for day in (1, 2):
                    keep = (target['subject'] == person) & (target['session'] == str(day)) & np.isin(target['y'], (1, 2))
                    values[a, o, j, day - 1] = ba(target['y'][keep] - 1, p[keep].argmax(1))
    rng = np.random.default_rng(352)
    draws = []
    for _ in range(10000):
        ps, ds, os = rng.integers(9, size=9), rng.integers(2, size=(9, 2)), rng.integers(8, size=8)
        draws.append(np.mean([values[:, os][:, :, p, ds[j]].mean((1, 2)) for j, p in enumerate(ps)], 0))
    draws = np.asarray(draws)
    mean = values.mean((1, 2, 3))
    result = {'scope': 'EXTERNAL BINARY CONDITION-DISCRIMINATION; not whole-system or physiological causality',
              'source_sanity': source, 'target_people': people_t, 'arms': ['power', 'cov'],
              'target_mean_ba': mean.tolist(), 'target_people_day_order_80ci': np.percentile(draws, [10, 90], 0).T.tolist(),
              'cov_minus_power_pp': float(mean[1] - mean[0]),
              'cov_minus_power_80ci': np.percentile(draws[:, 1] - draws[:, 0], [10, 90]).tolist(),
              'people_mean_ba': values.mean((1, 3)).tolist(), 'all_people_day_order_ba': values.tolist(),
              'pair_rows': int(np.isin(target['y'], (1, 2)).sum()), 'source_model_sha256': proof['source_model_sha256']}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('target_mean_ba', 'target_people_day_order_80ci', 'cov_minus_power_pp', 'cov_minus_power_80ci')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.out)
