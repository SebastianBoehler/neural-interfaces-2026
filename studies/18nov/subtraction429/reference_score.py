"""Paired reference sensitivity and actual v5.1b observation counts."""
import json
from pathlib import Path

import numpy as np

from wang_inputs import orders

PREVIOUS = Path('../ni26_wang429_v1_20261118')


def summarize(x, people, order):
    b = x[people[:,:,None], order].mean((1,2))
    return dict(mean=float(x.mean()), ci80=np.quantile(b,[.1,.9]).tolist())


def main():
    rng = np.random.default_rng(429)
    p = rng.integers(0,6,(10000,6))
    perm = rng.integers(0,2,(10000,6,2))
    results = {}
    for arm in ('old','cap','A6','pop'):
        by_reference = []
        counts = []
        for root in (PREVIOUS, Path('.')):
            people = []
            degenerate = 0
            for person in range(2,8):
                name = arm if arm in ('old','cap') else 'a6'
                records = json.loads((root/f'{name}_{person}.json').read_text())
                cells = np.full((2,3),np.nan)
                for r in records:
                    if r['arm'] != arm:
                        continue
                    visit, seed = r['visit'], r['seed']
                    j = (429,1429).index(seed)
                    rows = dict(orders(person,visit))[seed]
                    assert np.array_equal(rows,r['rows'])
                    pred = np.array(r['labels']) == 1
                    sub = rows < 6
                    cells[j,visit-1] = 100*(pred[sub].mean()-pred[~sub].mean())
                    if arm == 'old':
                        degenerate += r['activity']['degen']
                assert np.isfinite(cells).all()
                people.append(cells.mean(1))
            by_reference.append(np.array(people))
            counts.append(degenerate)
        out = summarize(by_reference[1]-by_reference[0],p,perm)
        if arm == 'old':
            assert counts == [648,0]
            out['degenerate_observations_original_and_average'] = counts
        results[arm] = out
    Path('reference_score.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
