"""Score paired whole archive and intervention endpoints for fresh orders."""
import json
from pathlib import Path

import numpy as np
from score import contrast, read


def main():
    arms = ('old','stock','zero','off')
    reference = {}
    scores = np.full((4,2,26,2), np.nan)
    with np.load('../neural-interfaces-2026/data/shin2018/windows.npz') as z:
        y, runs, subjects = z['y'], z['run'], z['subject']
    expected = {p:np.flatnonzero(subjects == f'Shin2018/{p}')[::2] for p in range(1,27)}
    for a,arm in enumerate(arms):
        records = sum((read(f'{arm}_{s}.json') for s in range(6)), [])
        assert read(f'{arm}_-1.json') == [r for r in records if r['person'] == 1]
        assert len(records) == 104
        for rec in records:
            p,seed,h = rec['person'],rec['seed'],rec['horizon']
            raw = expected[p][np.random.default_rng([415,p,seed,0]).permutation(60)]
            assert raw.tolist() == rec['raw']
            key = (p,seed,h)
            if arm == 'stock':
                reference[key] = rec['labels']
            if arm in ('zero','off'):
                n = 60 if h == 20 else 20
                assert rec['labels'][:n] == reference[key][:n]
            idx = a,(60,20).index(h),p-1,(2422,3422).index(seed)
            assert np.isnan(scores[idx])
            scores[idx] = contrast(rec['labels'],raw,y,runs)
    assert np.isfinite(scores).all()
    base = scores[1:] - scores[:1]
    stock,zero,off = base[:,0] - base[:,1]
    rng = np.random.default_rng(422)
    people = rng.integers(26,size=(10000,26))
    orders = rng.integers(2,size=(10000,26,2))
    results = {}
    for name,delta in [('stock',stock),('zero',zero),('off',off),('optimizer',stock-zero),('branch',zero-off)]:
        avg = delta.mean(1)
        results[name] = dict(mean=avg.mean(),ci80=np.quantile(avg[people].mean(1),[.1,.9]).tolist(),
            nested_ci80=np.quantile(delta[people[:,:,None],orders].mean((1,2)),[.1,.9]).tolist(),
            persons=avg.tolist(),orders=delta.mean(0).tolist())
    Path('result.json').write_text(json.dumps(dict(effects=results,
        absolute_selectivity=dict(zip(arms,scores.mean((2,3)).tolist())),horizon_order=[60,20]),indent=2)+'\n')


if __name__ == '__main__':
    main()
