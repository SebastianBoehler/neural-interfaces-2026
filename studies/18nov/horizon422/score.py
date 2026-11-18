"""Paired causal contrasts, using the original archive's identical external rows."""
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def contrast(labels, raw, y, run):
    labels = np.array(labels)
    x = (labels == 2).astype(int) - (labels == 1).astype(int)
    cells = []
    for g in np.unique(run[raw]):
        mask = run[raw] == g
        on, off = mask & (y[raw] == 1), mask & (y[raw] == 0)
        assert on.sum() == off.sum() == 10
        cells.append(100 * (x[on].mean() - x[off].mean()))
    assert len(cells) == 3
    return np.mean(cells)


def main():
    records = sum((read(f'run_{i}.json') for i in range(6)), [])
    reserve = read('run_-2.json')
    assert reserve == [r for r in records if r['person'] == 26]
    assert read('run_-1.json') == [r for r in records if r['person'] == 1]
    folders = ['../ni26_select415_20261118', '../ni26_short415_20261118']
    old = {}
    for horizon, folder, offset in zip((60, 20), folders, (6, 0)):
        for i in range(6):
            for r in read(f'{folder}/old_{offset+i}.json'):
                old[r['person'], r['seed'], horizon] = r
    assert len(old) == 104 and len(records) == 312
    values = np.full((4, 2, 26, 2), np.nan)
    with np.load('../neural-interfaces-2026/data/shin2018/windows.npz') as z:
        y, run = z['y'], z['run']
    for r in records:
        p, seed, h = r['person'], r['seed'], r['horizon']
        ref = old[p, seed, h]
        assert r['raw'] == ref['raw']
        idx = (0 if h == 60 else 1, p-1, (415, 1415).index(seed))
        arm = ('old', 'stock', 'zero', 'off').index(r['arm'])
        assert np.isnan(values[(arm, *idx)])
        for a, row in ((arm, r), (0, ref)):
            values[(a, *idx)] = contrast(row['labels'], row['raw'], y, run)
    assert np.isfinite(values).all()
    indexed = {(r['person'],r['seed'],r['horizon'],r['arm']): r for r in records}
    indices = sorted(old)
    changed = {}
    activity = {}
    for horizon in (60,20):
        keys = [k for k in indices if k[2] == horizon]
        for arm in ('stock','zero','off'):
            blocks = [b for k in keys for b in indexed[(*k,arm)]['activity']]
            activity[f'{arm}_{horizon}'] = {name: sum(b[name] for b in blocks) for name in
                ('windows','ft_refits','ft_steps','ft_relabels','cp_refits','cp_fused','errors')}
        for left,right in [('stock','zero'),('zero','off'),('stock','off')]:
            a,b = (np.array([indexed[(*k,arm)]['labels'] for k in keys]) for arm in (left,right))
            changed[f'{left}_{right}_{horizon}'] = (a != b).reshape(52,3,20).sum((0,2)).tolist()
    base = values[1:] - values[:1]
    stock, zero, off = base[:, 0] - base[:, 1]
    prior = read(f'{folders[1]}/result.json')
    np.testing.assert_allclose(stock.mean(1), prior['person'], atol=1e-10)
    rng = np.random.default_rng(422)
    people = rng.integers(26, size=(10000, 26))
    order = rng.integers(2, size=(10000, 26, 2))

    def estimate(x):
        person = x.mean(1)
        return dict(mean=x.mean(), ci80=np.quantile(person[people].mean(1), [.1,.9]).tolist(),
                    nested_ci80=np.quantile(x[people[:,:,None], order].mean((1,2)), [.1,.9]).tolist(),
                    orders=x.mean(0).tolist(), persons=person.tolist())
    effects = {k:estimate(v) for k,v in [('stock',stock), ('zero',zero), ('off',off),
                                          ('optimizer',stock-zero), ('branch',zero-off)]}
    np.testing.assert_allclose((stock-zero)+(zero-off)+off, stock)
    Path('result.json').write_text(json.dumps(dict(
        effects=effects, absolute_selectivity=dict(zip(('old','stock','zero','off'), values.mean((2,3)).tolist())),
        changed_per_20_deliveries=changed, activity=activity, horizon_order=[60,20], people=26, deliveries=2, reserve_person=26), indent=2)+'\n')


if __name__ == '__main__':
    main()
