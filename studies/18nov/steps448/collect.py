"""Guarded official external score and within-allocation paired resource use."""
import json
from pathlib import Path

import numpy as np
from data import permutation
from protocol import score

ARMS = ('stock', 'lower', 'zero', 'alternate')
PRIOR = Path('../ni26_qualify385_v1_20261118')


def read(path):
    return Path(path).read_text()


def compare(left, right, cells, seconds, predictions, draws):
    a, b = ARMS.index(left), ARMS.index(right)
    delta = (cells[..., a, :] - cells[..., b, :]).mean(-1)
    ratios = seconds[..., a].sum((1, 2)) / seconds[..., b].sum((1, 2))
    return dict(pp=float(delta.mean()), ci80=np.quantile(delta[draws].mean((1, 2, 3)), [.1, .9]).tolist(),
                person_pp=delta.mean((1, 2)).tolist(), order_pp=delta.mean((0, 1)).tolist(),
                median_time_ratio=float(np.median(ratios)), person_time_ratios=ratios.tolist(),
                decisions_changed=int((predictions[a] != predictions[b]).sum()))


def main():
    assert read('replay.txt').splitlines()[-1] == 'OK'
    with np.load('../neural-interfaces-2026/data/scherer2015/windows.npz') as archive:
        people, days, interface, y = (archive[k] for k in ('subject', 'session', 'interface', 'y'))
    sessions = np.char.add(np.char.add(people, '|'), days)
    orders = [permutation(sessions, None, 'full', s) for s in range(2)]
    cells = np.zeros((9, 2, 2, 4, 3))
    seconds = np.zeros(cells.shape[:-1])
    counts = np.zeros_like(seconds, dtype=int)
    steps = np.zeros_like(counts)
    predictions = np.full((4, 2, len(people)), -1)
    for i, person in enumerate(sorted(set(people))):
        runs = json.loads(read(f'person_{i}.json'))
        controls = json.loads(read(PRIOR / f'cap_{i}.json'))
        assert runs.keys() == controls.keys() and len(runs) == 4
        for d, day in enumerate(sorted(set(days[people == person]))):
            for s, order in enumerate(orders):
                key = f'{person}|{day}|{s}'
                block = runs[key]
                ref = controls[key]
                idx = order[(people == person) & (days == day)]
                assert idx.tolist() == block['rows'] == ref['rows']
                assert block['arms']['stock']['labels'] == ref['labels']
                for a, arm in enumerate(ARMS):
                    rec = block['arms'][arm]
                    labels = np.array(rec['labels'])
                    assert (predictions[a, s, idx] == -1).all()
                    predictions[a, s, idx] = labels
                    n = rec['activity']['ft_refits']
                    steps[i, d, s, a] = rec['activity']['ft_steps']
                    counts[i, d, s, a] = n
                    seconds[i, d, s, a] = rec['seconds']
                    assert rec['activity'] == dict(ref['activity'], ft_steps=int(steps[i, d, s, a]))
                    for c in range(3):
                        mask = y[idx] == c
                        assert mask.any()
                        cells[i, d, s, a, c] = 100 * (labels[mask] == c).mean()
    assert np.isin(predictions, (0, 1, 2)).all() and (seconds > 0).all()
    assert np.array_equal(steps[..., 1].sum(2), steps[..., 3].sum(2))
    assert np.array_equal(2 * steps[..., 0], 5 * steps[..., 1]) and not steps[..., 2].any()
    means = cells.mean((0, 1, 4))
    for a in range(4):
        for s in range(2):
            official = score(y, predictions[a, s], dict(stream=sessions, interface=interface), np.ones(len(y), bool))
            assert official['n_cells'] == 18
            np.testing.assert_allclose(100 * official['balanced_accuracy_cell_mean'], means[s, a], atol=1e-10, rtol=0)
    np.testing.assert_allclose(means.mean(0)[0], json.loads(read(PRIOR / 'result.json'))['acc'][1])
    rng = np.random.default_rng(448)
    draws = tuple(rng.integers(n, size=shape) for n, shape in
                  [(9, (10000, 9, 1, 1)), (2, (10000, 9, 2, 1)), (2, (10000, 9, 2, 2))])
    contrasts = {f'{a}-{b}': compare(a, b, cells, seconds, predictions, draws)
                 for a, b in [('lower', 'stock'), ('zero', 'stock'), ('lower', 'alternate'), ('alternate', 'stock')]}
    assert compare('stock', 'stock', cells, seconds, predictions, draws)['ci80'] == [0., 0.]
    result = dict(arms=ARMS, people=sorted(set(people)), accuracy=means.mean(0).tolist(),
                  class_recalls=cells.mean((0, 1, 2)).tolist(), by_order=means.tolist(), contrasts=contrasts,
                  refits=counts.sum((0, 1, 2)).tolist(), steps=steps.sum((0, 1, 2)).tolist(),
                  total_session_seconds=seconds.sum((0, 1, 2)).tolist(),
                  person_session_seconds=seconds.sum((1, 2)).tolist(), seconds=seconds.tolist())
    Path('result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
