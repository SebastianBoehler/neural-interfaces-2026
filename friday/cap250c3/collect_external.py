"""Score the locked full-system external decision bank."""
import json
from pathlib import Path

import numpy as np
from data import permutation, sha
from protocol import score

ARMS = ('stock', 'primary', 'placebo')
PRIOR = Path('../ni26_qualify385_v1_20261118')
EXTERNAL = Path('../neural-interfaces-2026/data/scherer2015/windows.npz')


def collect():
    with np.load(EXTERNAL, allow_pickle=False) as z:
        people, days, interfaces = (z[k] for k in ('subject', 'session', 'interface'))
    sessions = np.char.add(np.char.add(people, '|'), days)
    orders = [permutation(sessions, None, 'full', s) for s in (0, 1)]
    cohort = sorted(set(people))
    records, receipts = {}, {}
    predictions = np.full((3, 2, len(people)), -1)
    times = np.empty((3, 9, 2, 2))
    touched = np.zeros((3, 9), int)
    warm = {}
    for i, who in enumerate(cohort):
        prior = json.loads((PRIOR/f'cap_{i}.json').read_text())
        for a, arm in enumerate(ARMS):
            path = Path(f'{arm}_{i}.json')
            receipts[path.name] = sha(path)
            records[arm, who] = runs = json.loads(path.read_text())
            assert runs.keys() == prior.keys() and len(runs) == 4
            for d, day in enumerate(sorted(set(days[people == who]))):
                for k, order in enumerate(orders):
                    key = f'{who}|{day}|{k}'
                    rows = order[(people == who) & (days == day)]
                    run, expected = runs[key], prior[key]
                    assert rows.tolist() == run['rows'] == expected['rows']
                    assert run['activity'] == expected['activity']
                    if arm == 'stock':
                        assert run['labels'] == expected['labels']
                    assert (predictions[a, k, rows] == -1).all()
                    predictions[a, k, rows] = run['labels']
                    times[a, i, d, k] = run['seconds']
                    touched[a, i] += sum(e['changed'] for e in run['events'])
                    if 'warm_differences' in run:
                        warm[arm, who] = len(run['warm_differences'])
    assert np.isin(predictions, [0, 1, 2]).all()
    assert (times > 0).all() and len(warm) == 27
    assert not touched[0].any() and (touched[1] > 0).sum() >= 5
    Path('before_labels.json').write_text(json.dumps(receipts, indent=2))

    with np.load(EXTERNAL, allow_pickle=False) as z:
        labels = z['y']
    cells = np.empty((3, 9, 2, 2))
    for a in range(3):
        for i, who in enumerate(cohort):
            for d, day in enumerate(sorted(set(days[people == who]))):
                mask = (people == who) & (days == day)
                for k in (0, 1):
                    cells[a, i, d, k] = 100*np.mean([
                        (predictions[a, k, mask & (labels == c)] == c).mean()
                        for c in range(3)
                    ])
        for k in (0, 1):
            official = score(labels,predictions[a,k],dict(stream=sessions,interface=interfaces),
                             np.ones(len(labels),bool))
            assert official['n_cells'] == 18
            np.testing.assert_allclose(100*official['balanced_accuracy_cell_mean'],
                                       cells[a,:,:,k].mean(), atol=1e-10, rtol=0)
    baseline = json.loads((PRIOR/'result.json').read_text())['acc'][1]
    np.testing.assert_allclose(cells[0].mean(), baseline, atol=1e-9, rtol=0)
    rand = np.random.default_rng(463)
    draws = tuple(rand.integers(n, size=shape) for n, shape in (
        (9, (10000,9,1,1)), (2, (10000,9,2,1)), (2, (10000,9,2,2)),
    ))
    contrasts = {}
    for left, right in ((1,0),(1,2),(2,0)):
        delta = cells[left]-cells[right]
        arm, comparator = ARMS[left], ARMS[right]
        time_ratio = times[left].sum((1,2))/times[right].sum((1,2))
        hits = predictions[left] == labels
        previous = predictions[right] == labels
        contrasts[arm + '-' + comparator] = dict(
            pp=float(delta.mean()), paired80=np.quantile(delta[draws].mean((1,2,3)),[.1,.9]).tolist(),
            person_pp=delta.mean((1,2)).tolist(), order_pp=delta.mean((0,1)).tolist(),
            changed=int((predictions[left] != predictions[right]).sum()),
            corrections=int((hits & ~previous).sum()), regressions=int((previous & ~hits).sum()),
            median_time_ratio=float(np.median(time_ratio)), time_ratio=time_ratio.tolist(),
        )
    result = dict(people=cohort, accuracy=dict(zip(ARMS,cells.mean((1,2,3)).tolist())),
                  contrasts=contrasts, changed_weights=touched.tolist(),
                  warm_changes={'|'.join(k):v for k,v in warm.items()},
                  before_labels=sha(Path('before_labels.json')))
    Path('result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    collect()
