"""Descriptive individual uncertainty conditional on the three acquired runs."""
import json
from pathlib import Path

import numpy as np

from score import read


def cells(record, y, run):
    raw = record['raw']
    labels = np.array(record['labels'])
    signed = (labels == 2).astype(int) - (labels == 1).astype(int)
    result = []
    for name in sorted(set(run[raw])):
        mask = run[raw] == name
        on, off = mask & (y[raw] == 1), mask & (y[raw] == 0)
        assert on.sum() == off.sum() == 10
        result.append(100*(signed[on].mean() - signed[off].mean()))
    assert len(result) == 3
    return result


def main():
    recs = sum((read(f'run_{i}.json') for i in range(6)), [])
    for h,folder,offset in ((60,'../ni26_select415_20261118',6),(20,'../ni26_short415_20261118',0)):
        for i in range(6):
            for r in read(f'{folder}/old_{i+offset}.json'):
                recs.append(dict(**r,arm='old',horizon=h))
    with np.load('../neural-interfaces-2026/data/shin2018/windows.npz') as z:
        y, run = z['y'], z['run']
    v = np.full((4,2,26,2,3),np.nan)
    for r in recs:
        index = (('old','stock','zero','off').index(r['arm']), (60,20).index(r['horizon']),
                 r['person']-1, (415,1415).index(r['seed']))
        assert np.isnan(v[index]).all()
        v[index] = cells(r,y,run)
    assert np.isfinite(v).all()
    rel = v[1:] - v[:1]
    stock,zero,off = rel[:,0] - rel[:,1]
    # All 3^3 equally probable resamples of the three runs, keeping the two deliveries paired.
    draws = np.indices((3,3,3)).reshape(3,-1).T
    prior = read('result.json')['effects']
    report = {}
    for name,effect in [('stock',stock),('optimizer',stock-zero),('branch',zero-off),('off',off)]:
        order_means = effect.mean(2)
        np.testing.assert_allclose(order_means.mean(1), prior[name]['persons'], atol=1e-10)
        by_run = effect.mean(1)
        low,high = np.quantile(by_run[:,draws].mean(2),[.1,.9],axis=1)
        report[name] = [dict(person=i+1,orders=orders.tolist(),run_ci80=[lo,hi])
                        for i,(orders,lo,hi) in enumerate(zip(order_means,low,high))]
    Path('person.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
