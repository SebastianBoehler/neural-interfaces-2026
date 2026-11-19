"""Compare locked decisions with official class/cell weights and paired resampling."""
import json
from pathlib import Path

import numpy as np
import score379 as ref
from measure import official

from produce import CANDIDATES, public_inputs

ROLES = {'new': ref.PEOPLE, 'returning': ('P04', 'P10')}
GROUPS = (np.arange(4), np.arange(4, 6), np.arange(6, 8))
CONTRASTS = (('primary', 'stock'), ('primary', 'confidence'), ('history', 'primary'),
             ('history', 'stock'), ('confidence', 'stock'))


def interval(delta, weights):
    rng = np.random.default_rng(443)
    people_mean = sum(delta[:, g].mean(1) for g in GROUPS)/3
    people = rng.integers(len(weights), size=(4000, len(weights)))
    draws = np.zeros(people.shape)
    for group in GROUPS:
        orders = rng.choice(group, size=(*people.shape, len(group)))
        draws += delta[people[:, :, None], orders].mean(2)/3
    sampled = weights[people]
    bootstrap = (sampled*draws).sum(1)/sampled.sum(1)
    return dict(pp=float(np.average(people_mean, weights=weights)),
                paired80=np.percentile(bootstrap, [10, 90]).tolist(),
                people=people_mean.tolist(),
                orders=dict(zip(ref.ORDERS, np.average(delta, axis=0, weights=weights).tolist())))


def decisions():
    source = ref.sha(Path('source.sha256'))
    receipts, records = {}, {}
    for role in ROLES:
        for order in ref.ORDERS:
            path = Path(f'{role}_{order}.npz')
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert receipt['role'] == role and receipt['order'] == order
            assert receipt['output'] == ref.sha(path) and receipt['source'] == source
            with np.load(path) as z:
                records[role, order] = dict(z)
            receipts[path.stem] = receipt
    Path('before_labels.json').write_text(json.dumps(receipts, indent=2))
    return records


def score():
    records = decisions()
    meta, _ = public_inputs()
    with np.load(ref.BASE/'data/proteus_windows.npz') as z:
        labels = z['y']
    metric = official()
    roles = {}
    arms = ('stock', *CANDIDATES)
    for role, people in ROLES.items():
        scores = {a: np.zeros((len(people), len(ref.ORDERS))) for a in arms}
        weights = np.array([.2, .8]) if role == 'returning' else np.zeros(len(people))
        eligible = 0
        changes = {a+'-'+b: dict(changed=0, corrections=0, regressions=0) for a, b in CONTRASTS}
        role_mask = np.ones(len(labels), bool) if role == 'new' else (
            np.isin(meta['person'], people) & (meta['session'].astype(str) != '1')
        )
        rows = np.flatnonzero(role_mask)
        assert len(rows) == (2655 if role == 'new' else 762)
        for j, order in enumerate(ref.ORDERS):
            bank = records[role, order]
            assert np.array_equal(bank['rows'], rows)
            eligible += int(bank['eligible'].sum())
            for a, b in CONTRASTS:
                delta = changes[a+'-'+b]
                assert np.array_equal(bank[a][~bank['eligible']], bank[b][~bank['eligible']])
                right_a, right_b = bank[a] == labels[rows], bank[b] == labels[rows]
                delta['changed'] += int(np.count_nonzero(bank[a] != bank[b]))
                delta['corrections'] += int(np.count_nonzero(right_a & ~right_b))
                delta['regressions'] += int(np.count_nonzero(right_b & ~right_a))
            for i, person in enumerate(people):
                mask = role_mask & (meta['person'] == person)
                for a in arms:
                    result = metric(labels[mask], bank[a][mask[rows]], meta, mask)
                    if role == 'new':
                        weights[i] = result['n_cells']
                    else:
                        assert result['n_cells'] == 4
                    scores[a][i, j] = 100*result['balanced_accuracy_cell_mean']
        if role == 'new':
            assert weights.sum() == 28
        absolute = {a: interval(v, weights) for a, v in scores.items()}
        if role == 'new':
            assert abs(absolute['stock']['pp']-76.103915)<1e-5
        else:
            assert abs(absolute['stock']['pp']-86.574)<.001
        roles[role] = dict(eligible=eligible, total=len(rows)*len(ref.ORDERS), people=list(people),
                           absolute=absolute, contrasts={
            a+'-'+b: {**interval(scores[a]-scores[b], weights), **changes[a+'-'+b]}
            for a, b in CONTRASTS
        })
    joint = {}
    for a, b in CONTRASTS:
        key = a+'-'+b
        joint[key] = .566*roles['new']['contrasts'][key]['pp'] + .434*roles['returning']['contrasts'][key]['pp']
    Path('result.json').write_text(json.dumps(dict(roles=roles, joint_pp=joint,
                           receipt=ref.sha(Path('before_labels.json'))), indent=2))


if __name__ == '__main__':
    score()
