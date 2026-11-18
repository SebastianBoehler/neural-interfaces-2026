"""Recover paired responses from delivered decisions; bootstrap people and orders."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

from exposure import verify
from prepare import PEOPLE, digest


def main(repo):
    assert Path('replay.txt').read_text().splitlines()[-1] == 'OK'
    prior = verify(repo, PEOPLE)
    # Person, order, repeat, package, instructed paradigm, instructed hand.
    rates = np.full((6, 2, 2, 2, 2, 3), np.nan)
    activity = [Counter(), Counter()]
    changed = 0
    source = {}
    for person_index, person in enumerate(PEOPLE):
        src = json.loads(Path(f'source_{person}.json').read_text())
        source.update(src)
        logs = [json.loads(Path(f'{arm}_{person_index}.json').read_text()) for arm in ('old','cap')]
        assert all(len(log) == 8 for log in logs)
        for repeat in (1, 2):
            for condition, name in enumerate(('MI','ME')):
                key = f'S{person}_{name}{repeat}'
                path = Path(key + '.npz')
                assert digest(path) == src[key]['windows']
                with np.load(path) as bank:
                    hand = bank['label']
                    assert len(np.unique(bank['cue'])) == 30
                for order, seed in enumerate((447,1447)):
                    label_pair = []
                    for arm, data in enumerate(logs):
                        stream = data[f'{key}-{seed}']
                        rows = np.asarray(stream['rows'])
                        expected = np.random.default_rng([447,person,repeat,seed]).permutation(np.arange(30))
                        assert np.array_equal(rows, expected) and set(rows) == set(range(30))
                        pred = np.asarray(stream['labels'])
                        assert pred.shape == (30,) and set(pred) <= {0,1,2}
                        for h in range(3):
                            which = hand[rows] == h+1
                            assert sum(which) == 10
                            rates[person_index,order,repeat-1,arm,condition,h] = 100*np.mean(pred[which] == 0)
                        label_pair.append(pred)
                        signals = stream['activity']
                        for item, v in signals.items():
                            if item == 'recognized':
                                activity[arm]['recognized_streams'] += int(v >= 0)
                            elif isinstance(v, int):
                                activity[arm][item] += v
                        if person_index == order == repeat-1 == 0:
                            smoke = json.loads(Path(f'{("old","cap")[arm]}_-1.json').read_text())
                            assert stream == smoke[f'{key}-{seed}']
                    changed += int(np.sum(label_pair[0] != label_pair[1]))
    assert np.isfinite(rates).all()
    rng = np.random.default_rng(447)
    p = rng.integers(6, size=(16000,6,1))
    o = rng.integers(2, size=(16000,6,2))
    def interval(values):
        return [round(float(a), 4) for a in [values.mean(), *np.quantile(values[p,o].mean((1,2)), [.1,.9])]]
    comparisons = {}
    for visit in range(2):
        for h, hand_name in enumerate(('left','right','both')):
            effects = rates[:,:,visit,:,0,h] - rates[:,:,visit,:,1,h]
            comparisons[f'{visit+1}-{hand_name}'] = dict(
                old=interval(effects[:,:,0]), cap=interval(effects[:,:,1]),
                difference=interval(effects[:,:,1]-effects[:,:,0]),
                absolute=rates[:,:,visit,:,:,h].mean((0,1)).tolist(),
                people=(effects[:,:,1]-effects[:,:,0]).mean(1).tolist(),
                orders=(effects[:,:,1]-effects[:,:,0]).mean(0).tolist())
    result = dict(comparisons=comparisons, people=PEOPLE, previously_exported=prior,
                  changed=changed, unique_rows=720, paired_decisions=1440,
                  old_activity=dict(activity[0]),cap_activity=dict(activity[1]),sources=source)
    Path('result.json').write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
