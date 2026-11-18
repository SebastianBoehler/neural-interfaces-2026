"""Evaluate frozen two-seed memory corrections on the authenticated §443 system."""
import json
from pathlib import Path
import sys

import numpy as np
import score379 as ref
from fusion_inputs import official, probability, restore
from score import interval

HISTORY = ref.WORK / 'ni26_history443_20261118'
PINS = {379: 'd3124337ec0961838f72505a91158426fc000dbdcbdbb0ccc67121204bc9aa5d',
        1441: '0a7ceacc551c095110c9f0ea96868af41b57e1a5b0542ff6461753a6eb9ef5d2'}
CONTRASTS = {
    'agreement': [('consensus', 'stock'), ('consensus', 'mean'), ('consensus', 'donor_consensus')],
    'residual': [('residual', 'consensus'), ('residual', 'stock'),
                 ('residual', 'donor_residual'), ('residual', 'prior')],
}


def seed_deltas(seed, person, rows):
    result = HISTORY / f'{seed}_result.json'
    assert ref.sha(result) == PINS[seed]
    proof = json.loads(result.read_text())
    path = HISTORY / f'{seed}_{person}.npz'
    record = path.with_suffix('.json')
    assert ref.sha(record) == proof['actors'][person]
    assert ref.sha(path) == json.loads(record.read_text())['output']
    assert ref.source_manifest(HISTORY / 'source.sha256') == proof['source']
    archived = HISTORY / f'{seed}_decisions.npz'
    assert ref.sha(archived) == proof['predictions']
    with np.load(path) as z:
        assert np.array_equal(z['index'], rows)
        source = z['source'].astype(np.float64)
        corrections = {}
        for arm in ('memory', 'donor'):
            d = z[arm].astype(np.float64) - source
            assert np.isfinite(d).all()
            corrections[arm] = np.clip(d - d.mean(2, keepdims=True), -1, 1)
    with np.load(archived) as z:
        reference = {(order, arm): z[f'{person}|control|{order}|{arm}']
                     for order in ref.ORDERS for arm in ('stock', 'memory', 'donor')}
    return corrections, reference


def consensus(a, b):
    return np.where(a*b > 0, np.sign(a)*np.minimum(abs(a), abs(b)), 0)


def prior_mean(delta, ages, streams):
    prior = np.zeros_like(delta)
    for stream in np.unique(streams):
        rows = np.flatnonzero(streams == stream)
        order = rows[np.argsort(ages[rows])]
        assert np.array_equal(ages[order], np.arange(len(order)))
        assert np.all(delta[order[0]] == 0)
        prior[order[1:]] = np.cumsum(delta[order[:-1]], axis=0)/(20+ages[order[1:], None])
        assert np.all(prior[order[0]] == 0)
    return prior


def build(phase, people):
    sc, scores, meta, ages = restore()
    controls, verified_age = ref.controls_and_age(meta)
    assert tuple(sc.ORDERS) == ref.ORDERS
    bank = {}
    for person in people:
        rows = np.flatnonzero(meta['person'] == person)
        inputs = [seed_deltas(seed, person, rows) for seed in PINS]
        for i, order in enumerate(ref.ORDERS):
            age = ages[order][rows]
            assert np.array_equal(age, verified_age[i, rows])
            prob = probability(scores['control', order, 'B2_pool'][rows], age)
            with np.errstate(divide='ignore'):
                base = np.log(prob)
            stock = base.argmax(1)
            assert np.array_equal(stock, controls['frozen'][i, rows])
            arms = {'stock': np.zeros_like(prob)}
            for name in ('memory', 'donor'):
                a, b = [corr[name][i] for corr, _ in inputs]
                for (_, reference), d in zip(inputs, (a, b)):
                    assert np.array_equal(stock, reference[order, 'stock'])
                    assert np.array_equal((base+.1*d).argmax(1), reference[order, name])
                prefix = '' if name == 'memory' else 'donor_'
                arms[prefix+'consensus'] = consensus(a, b)
                if phase == 'agreement':
                    arms[prefix+'mean'] = (a+b)/2
                else:
                    pa, pb = [prior_mean(d, age, meta['stream'][rows]) for d in (a, b)]
                    arms[prefix+'prior'] = consensus(pa, pb)
                    arms[prefix+'residual'] = consensus(np.clip(a-pa, -1, 1), np.clip(b-pb, -1, 1))
            for arm, d in arms.items():
                assert np.isfinite(d).all() and np.all(abs(d) <= 1)
                pred = (base+.1*d).argmax(1)
                if arm == 'stock':
                    assert np.array_equal(pred, stock)
                bank[person, order, arm] = pred.astype(np.uint8)
    return meta, bank


def score(phase):
    meta, bank = build(phase, ref.PEOPLE)
    out = Path(phase+'_decisions.npz')
    np.savez_compressed(out, **{'|'.join(k): v for k, v in bank.items()})
    provenance = dict(predictions=ref.sha(out), source=ref.sha(Path('source.sha256')), parent=PINS)
    Path(phase+'_before_labels.json').write_text(json.dumps(provenance, indent=2))
    assert ref.sha(ref.BASE/'data/proteus_windows.npz') == ref.DATA_SHA
    with np.load(ref.BASE/'data/proteus_windows.npz') as z:
        y = z['y']
    metric = official()
    scores = {k[2]: np.empty((10, 8)) for k in bank}
    changes = {arm: 0 for arm in scores}
    counts = []
    for j, person in enumerate(ref.PEOPLE):
        mask = meta['person'] == person
        counts.append(len(set(zip(meta['stream'][mask], meta['interface'][mask]))))
        for i, order in enumerate(ref.ORDERS):
            stock = bank[person, order, 'stock']
            for arm in scores:
                pred = bank[person, order, arm]
                changes[arm] += int((pred != stock).sum())
                entry = metric(y[mask], pred, meta, mask)
                assert entry['n_cells'] == counts[-1]
                scores[arm][j, i] = 100*entry['balanced_accuracy_cell_mean']
    counts = np.array(counts)
    assert counts.sum() == 28 and all(changes[a] for a in changes if a != 'stock')
    absolute = {arm: interval(values, counts)['pp'] for arm, values in scores.items()}
    assert abs(absolute['stock']-76.2793) < .00005
    contrasts = {}
    for a, b in CONTRASTS[phase]:
        delta = scores[a]-scores[b]
        contrasts[a+'-'+b] = dict(**interval(delta, counts),
                                  orders=dict(zip(ref.ORDERS, np.average(delta, axis=0, weights=counts))))
    Path(phase+'_result.json').write_text(json.dumps(dict(**provenance, absolute=absolute, changed=changes,
                                                          contrasts=contrasts), indent=2))


if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        for wave in CONTRASTS:
            build(wave, ('P01',))
    else:
        score(sys.argv[1])
