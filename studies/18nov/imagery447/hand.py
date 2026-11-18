"""Difference of hand contrasts cancels a common file-wide response shift."""
import json
from pathlib import Path

import numpy as np

PEOPLE = (211, 216, 221, 222, 226, 227)


def main():
    primary = json.loads(Path('result.json').read_text())['comparisons']
    rng = np.random.default_rng(1447)
    persons = rng.integers(6, size=(16000, 6, 1))
    orders = rng.integers(2, size=(16000, 6, 2))
    result = {}
    for visit in (1, 2):
        # Ordered percentage-point [imagery - execution] per hand.
        rates = np.zeros((6, 2, 2, 3))
        for i, person in enumerate(PEOPLE):
            for a, arm in enumerate(('old', 'cap')):
                logs = json.loads(Path(f'{arm}_{i}.json').read_text())
                for task, sign in [('MI', 1), ('ME', -1)]:
                    key = f'S{person}_{task}{visit}'
                    with np.load(key + '.npz') as src:
                        hand = src['label']
                    for j, seed in enumerate((447, 1447)):
                        log = logs[f'{key}-{seed}']
                        values = np.zeros(30, dtype=bool)
                        values[log['rows']] = np.equal(log['labels'], 0)
                        for h in range(3):
                            rates[i,j,a,h] += sign * 100 * values[hand == h+1].mean()
        for h, name in enumerate(('left','right','both')):
            assert np.allclose(rates[:,:,:,h].mean((0,1)),
                               [primary[f'{visit}-{name}'][a][0] for a in ('old','cap')], atol=0.0001)
        effects = rates[:,:,:,0] - rates[:,:,:,1]
        summaries = {}
        for name, d in [('old',effects[:,:,0]), ('cap',effects[:,:,1]),
                        ('difference',effects[:,:,1]-effects[:,:,0])]:
            boot = d[persons,orders].mean((1,2))
            summaries[name] = dict(mean=d.mean(), interval=np.quantile(boot,[.1,.9]).tolist(),
                                   people=d.mean(1).tolist())
        result[str(visit)] = summaries
    Path('hand.json').write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
