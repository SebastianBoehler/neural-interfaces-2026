"""Paired within-run task selectivity on frozen native package decisions."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np

DATA = Path('../neural-interfaces-2026/data')


def key(row):
    return row['kind'], row['person'], row['session'], row['seed']


def main():
    predictions = {}
    trace = hashlib.sha256()
    for arm in ('old', 'cap'):
        records = {}
        for shard in range(12):
            file = Path(f'{arm}_{shard}.json')
            trace.update(file.read_bytes())
            for r in json.loads(file.read_text()):
                assert key(r) not in records and len(r['raw']) == len(r['labels'])
                records[key(r)] = r
        predictions[arm] = records
    assert predictions['old'].keys() == predictions['cap'].keys()
    assert len(predictions['old']) == 2 * (29 * 3 + 26)
    by_person, activity = defaultdict(list), defaultdict(list)
    for k, old in predictions['old'].items():
        new = predictions['cap'][k]
        assert all(old[v] == new[v] for v in ('filename', 'rows', 'raw', 'groups'))
        kind, person, session, seed = k
        folder = 'shin2017b' if kind == 'A' else 'shin2018'
        with np.load(DATA / folder / old['filename']) as z:
            subject = ('Shin2017B/' if kind == 'A' else 'Shin2018/') + str(person)
            relevant = np.flatnonzero(z['subject'] == subject)
            ss, runs, onsets = (z[name][relevant] for name in ('session', 'run', 'onset'))
            tags = np.char.add(np.char.add(ss, '|'), runs)
            first = []
            for tag in sorted(set(tags)):
                source = np.flatnonzero(tags == tag)
                # Select earlier crops independently by matching their onsets to a later crop.
                earlier = np.isclose(onsets[source, None] + 4, onsets[source][None, :], atol=1e-6, rtol=0).any(1)
                assert earlier.sum() == 20
                first.extend(source[earlier])
            first = np.asarray(sorted(first))
            si = sorted(set(ss[first])).index(session)
            local = np.flatnonzero(ss[first] == session)
            offsets = np.random.default_rng([415, person, seed, si]).permutation(local)
            rows = relevant[first[offsets]]
            assert np.array_equal(rows, old['raw']) and np.array_equal(offsets, old['rows'])
            assert np.array_equal(tags[first[offsets]], old['groups'])
            y = z['y'][rows]
        groups = np.array(old['groups'])
        pred = np.asarray([old['labels'], new['labels']])
        assert np.isin(pred, [0, 1, 2]).all()
        for group in sorted(set(groups)):
            mask = group == groups
            assert sorted(np.bincount(y[mask])) == [10, 10]
            rates = np.array([[(pred[:, mask & (y == active)] == c).mean(1)
                               for c in range(3)] for active in [0, 1]])
            by_person[kind, person].append(rates)
        activity[kind].append(new['activity'])
    results = {}
    for kind, target in [('A', 1), ('W', 2)]:
        persons = sorted(p for kind0, p in by_person if kind0 == kind)
        assert persons == list(range(1, (29 if kind == 'A' else 26) + 1))
        assert all(len(by_person[kind, p]) == 6 for p in persons)
        # person × rest/active × decoded class × archive
        rates = 100 * np.array([np.mean(by_person[kind, p], axis=0) for p in persons])
        change = rates[:, 1] - rates[:, 0]
        contrast = change[:, target] - change[:, 3-target]
        delta = contrast[:, 1] - contrast[:, 0]
        boot = np.random.default_rng(415).choice(delta, (10000, len(persons))).mean(1)
        act = activity[kind]
        results[kind] = dict(n=len(persons), rates=rates.mean(0).tolist(),
                             selectivity=contrast.mean(0).tolist(), delta=float(delta.mean()),
                             ci80=np.quantile(boot, [.1, .9]).tolist(), per_person=delta.tolist(),
                             sessions_with_order=len(act), recognized=sum(a['recognized'] >= 0 for a in act),
                             refits=sum(a['ft_refits'] for a in act), steps=sum(a['ft_steps'] for a in act))
    results['prediction_sha256'] = trace.hexdigest()
    Path('result.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
