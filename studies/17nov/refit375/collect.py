"""Score fixed causal decisions and label-assisted look-ahead separately."""
import json
from pathlib import Path

import numpy as np

from trajectory import INPUT, check, digest, prepare

ARMS = ('fixed','delayed','drift','short_oracle','full_oracle')


def main():
    assets, _, cfg = prepare()
    scorer, _, people, _, meta, _ = assets.load_controls(dtype=np.float32)
    _, metric = __import__('run').owner()
    returning = assets.returning_guard(scorer.ORDERS, cfg['v6_cp_w0'])
    shards = [Path(f'shard_{i}.npz') for i in range(20)]
    before = dict(shards={p.name:digest(p) for p in shards},source=digest('source.sha256'),
                  smoke=digest('smoke.json'),returning_stops=returning)
    payload = {}
    for path in shards:
        with np.load(path) as shard:
            check(not payload.keys() & set(shard.files), 'duplicate sessions')
            payload.update({k:shard[k] for k in shard.files})
    check(len(payload) == 8*14*4, 'missing archive')
    with np.load(INPUT/'inputs.npz') as control:
        for order in scorer.ORDERS:
            idx, bounds = (control[order+'|'+name] for name in ('index','bounds'))
            for sid,rows in enumerate(np.split(idx,bounds[1:-1])):
                check(np.array_equal(payload[f'{order}|{sid}|rows'],rows), 'order mismatch')
                check(np.array_equal(payload[f'{order}|{sid}|fixed'],control[order+'|expected'][rows].argmax(1)), 'control labels')
                check(len(set(meta['stream'][rows])) == 1, 'session boundary')
    Path('before_labels.json').write_text(json.dumps(before,indent=2))
    with np.load(assets.CHAIN/'data/proteus_windows.npz') as z:
        y = z['y']
    cells = np.char.add(np.char.add(meta['stream'],'|'),meta['interface'])
    weights = np.zeros(len(y))
    for cell in np.unique(cells):
        for task in range(3):
            mask = (cells == cell) & (y == task)
            check(mask.any(),'missing class')
            weights[mask] = 1/mask.sum()
    values = {arm:np.zeros((8,10)) for arm in ARMS}
    counts = np.zeros(10,dtype=int)
    decisions = np.zeros((8,3),dtype=int)
    for oi,order in enumerate(scorer.ORDERS):
        outputs = {arm:np.full(len(y),-1) for arm in ARMS}
        for sid in range(14):
            rows,fixed,delayed,drift = (payload[f'{order}|{sid}|'+k] for k in ('rows','fixed','delayed','drift'))
            gain = weights[rows]*((delayed == y[rows]).astype(int) - (fixed == y[rows]).astype(int))
            choose = [drift <= .1, gain[100:110].sum() > 0, gain.sum() > 0]
            decisions[oi] += choose
            for name,labels in zip(ARMS,[fixed,delayed,*[delayed if switch else fixed for switch in choose]],strict=True):
                outputs[name][rows] = labels
        for pi,person in enumerate(scorer.PEOPLE):
            mask = people == person
            for arm in ARMS:
                row = metric(y[mask],outputs[arm][mask],meta,mask)
                counts[pi] = row['n_cells']
                values[arm][oi,pi] = 100*row['balanced_accuracy_cell_mean']
    for a,b in [('drift','fixed'),('delayed','fixed'),('full_oracle','fixed'),('full_oracle','short_oracle')]:
        values[a+'-'+b] = values[a]-values[b]
    results = {a:dict(official=scorer.cell_weighted_summary(v,scorer.ORDERS,counts),
                      people_order=scorer.summarize(v,scorer.ORDERS,scorer.PEOPLE))
               for a,v in values.items()}
    check(abs(results['fixed']['official']['mean']-76.27933502145758)<1e-7,'full baseline')
    check(values['full_oracle-short_oracle'].min()>-1e-10,'oracle monotonicity')
    Path('result.json').write_text(json.dumps(dict(results=results,delayed_sessions=decisions.tolist(),
        orders=scorer.ORDERS,source=digest('source.sha256')),indent=2))


if __name__ == '__main__':
    main()
