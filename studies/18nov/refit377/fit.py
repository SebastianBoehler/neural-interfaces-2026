"""Learn source-only decisions before evaluating the outer person."""
import json
from pathlib import Path

import numpy as np

from produce import PEOPLE, ORDERS, check, digest, prepare, verify
from run import owner

ARMS = ('fixed', 'delayed', 'short', 'full', 'short_mean', 'full_mean')
ORDER_W = np.array([1/12] * 4 + [1/6] * 4)


def source_reward(data, y, meta):
    rewards = []
    for order in ORDERS:
        rows = data[order+'|rows']
        fixed, delay = data[order+'|proxy']
        cells = meta['interface'][rows]
        labels = y[rows]
        weights = np.zeros(len(rows))
        for cell in np.unique(cells):
            for task in range(3):
                select = (cells == cell) & (labels == task)
                check(select.any(), 'source cell missing class')
                weights[select] = 1/(select.sum()*3*len(np.unique(cells)))
        gain = weights*((delay == labels).astype(int) - (fixed == labels).astype(int))
        rewards.append([gain[100:110].sum(), gain.sum()])
    return np.array(rewards)


def fit_fold(held, matrix, meta, path):
    train = [p for p in PEOPLE if p != held]
    rows = np.concatenate([np.unique(matrix[p][ORDERS[0]+'|rows']) for p in train])
    check(not set(rows) & set(matrix[held][ORDERS[0]+'|rows']), 'outer train overlap')
    with np.load(path) as z:
        y = np.full(len(meta['stream']), -1)
        y[rows] = z['y'][rows]
    design = np.stack([matrix[p][o+'|proxy_features'] for p in train for o in ORDERS])
    rewards = np.concatenate([source_reward(matrix[p], y, meta) for p in train])
    weights = np.tile(ORDER_W, len(train))/len(train)
    center = weights @ design
    scale = np.sqrt(weights @ ((design - center)**2))
    check((scale > 1e-8).all() and np.isclose(weights.sum(), 1), 'ridge scaling')
    x = (design-center)/scale
    mean = weights @ rewards
    coef = np.linalg.solve(x.T @ (weights[:, None]*x) + np.eye(3), x.T @ (weights[:, None]*(rewards-mean)))
    actions = {}
    forecasts = {}
    for kind in ('proxy', 'target'):
        v = np.stack([matrix[held][o+'|'+kind+'_features'] for o in ORDERS])
        pred = (v-center)/scale @ coef + mean
        forecasts[kind] = pred.tolist()
        actions[kind] = np.column_stack([np.zeros(8), np.ones(8), pred > 0, np.tile(mean > 0,(8,1))]).astype(int).tolist()
    return dict(held=held, train=train, scale=scale.tolist(), coef=coef.tolist(), intercept=mean.tolist(),
                forecasts=forecasts, actions=actions)


def main():
    verify()
    assets, _, cfg = prepare()
    scorer, _, _, _, meta, _ = assets.load_controls(dtype=np.float32)
    _, metric = owner()
    check(tuple(scorer.ORDERS) == ORDERS, 'order contract')
    path = assets.CHAIN/'data/proteus_windows.npz'
    matrix = {}
    for person in PEOPLE:
        with np.load(person+'.npz') as z:
            matrix[person] = {k:z[k] for k in z.files}
    records = {p:fit_fold(p,matrix,meta,path) for p in PEOPLE}
    proof = dict(folds=records, source=digest('source.sha256'), matrices={p:digest(p+'.npz') for p in PEOPLE},
                 replay=Path('replay.log').read_text().splitlines()[-1],
                 returning_stops=assets.returning_guard(ORDERS,cfg['v6_cp_w0']))
    check(proof['replay'] == 'OK', 'historical v3 guard')
    Path('actions.json').write_text(json.dumps(proof,indent=2))
    results = {}
    with np.load(path) as z:
        y = z['y']
    for kind in ('proxy','target'):
        vals = {arm:np.zeros((8,4)) for arm in ARMS}
        counts = np.zeros(4, dtype=int)
        for pi,person in enumerate(PEOPLE):
            data = matrix[person]
            for oi,order in enumerate(ORDERS):
                rows = data[order+'|rows']
                mask = np.isin(np.arange(len(y)), rows)
                arr = data[order+'|'+kind][:, np.argsort(rows)]
                for ai,arm in enumerate(ARMS):
                    choice = records[person]['actions'][kind][oi][ai]
                    m = metric(y[mask],arr[choice],meta,mask)
                    vals[arm][oi,pi] = 100*m['balanced_accuracy_cell_mean']
                    counts[pi] = m['n_cells']
        for a,b in [('full','short'),('full','fixed'),('full','full_mean'),('short','short_mean'),('delayed','fixed')]:
            vals[a+'-'+b] = vals[a]-vals[b]
        results[kind] = {name:dict(official=scorer.cell_weighted_summary(v,ORDERS,counts),
                                    people_order=scorer.summarize(v,ORDERS,PEOPLE),
                                    people=np.round(ORDER_W @ v,6).tolist()) for name,v in vals.items()}
    Path('result.json').write_text(json.dumps(dict(results=results, actions_sha256=digest('actions.json')), indent=2))


if __name__ == '__main__':
    main()
