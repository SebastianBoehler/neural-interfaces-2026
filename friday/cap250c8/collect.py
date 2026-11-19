"""Authenticate the worker banks and compute paired metric and order replication."""
import json
from pathlib import Path

import numpy as np
from data import permutation,sha
from protocol import score

PROBE = Path('../neural-interfaces-2026/data/scherer2015/windows.npz')
PRIOR = Path('../ni26_cross411b_20261118')
ORDERS = (411, 1411)
ARMS = ('stock','inner','causal')


def decisions(people,days,groups):
    roster = sorted(set(people))
    orders = {s:permutation(groups,None,'full',s) for s in ORDERS}
    source = sha(Path('source.sha256'))
    predictions = np.full((len(ARMS),2,len(people)),-1)
    receipts={}
    for i,person in enumerate(roster):
        path = Path(f'held_{i}.json')
        assert sha(path) == path.with_suffix('.sha256').read_text().strip()
        capture = json.loads(path.read_text())
        assert capture['source'] == source
        cap,old = (json.loads((PRIOR/f'{a}_{i}.json').read_text()) for a in ('cap','old'))
        assert capture['sessions'].keys() == cap.keys() == old.keys() and len(cap) == 4
        for name,rec in capture['sessions'].items():
            who, day, order = name.split('|')
            order = int(order)
            j = ORDERS.index(order)
            rows = orders[order][(people==person)&(days==day)]
            assert who == person and rows.tolist()==rec['rows']==cap[name]['rows']==old[name]['rows']
            assert rec['activity']==cap[name]['activity']
            v=rec['predictions']
            assert set(v)==set(ARMS) and (predictions[:,j,rows]==-1).all()
            assert v['stock']==cap[name]['labels'] and v['inner']==old[name]['labels']
            assert np.isin(v['causal'],(0,1,2)).all()
            assert all(x in (a,b) for x,a,b in zip(v['causal'],v['stock'],v['inner']))
            for a,arm in enumerate(ARMS):
                predictions[a,j,rows]=v[arm]
        receipts[path.name]=sha(path)
    assert np.isin(predictions,(0,1,2)).all()
    Path('before_labels.json').write_text(json.dumps(receipts,indent=2)+'\n')
    return roster,predictions


def collect():
    with np.load(PROBE,allow_pickle=False) as z:
        people,days,interface=(z[k] for k in ('subject','session','interface'))
    groups=np.char.add(np.char.add(people,'|'),days)
    roster,pred=decisions(people,days,groups)
    with np.load(PROBE,allow_pickle=False) as z:
        y=z['y']
    cells=np.empty((len(ARMS),len(roster),2,2))
    for a,arm in enumerate(ARMS):
        for i,person in enumerate(roster):
            for d,day in enumerate(sorted(set(days[people==person]))):
                m=(people==person)&(days==day)
                assert np.unique(y[m]).tolist()==[0,1,2]
                for order in (0,1):
                    cells[a,i,d,order]=100*np.mean([
                        (pred[a,order,m&(y==c)]==c).mean() for c in range(3)])
        for j in (0,1):
            official=score(y,pred[a,j],dict(stream=groups,interface=interface),np.ones(len(y),bool))
            assert official['n_cells']==18
            np.testing.assert_allclose(100*official['balanced_accuracy_cell_mean'],
                                       cells[a,:,:,j].mean(),atol=1e-10,rtol=0)
    anchor=json.loads((PRIOR/'result.json').read_text())['acc']
    np.testing.assert_allclose(cells[:2].mean((1,2,3)),anchor[::-1],atol=1e-10,rtol=0)
    gap=cells[2]-cells[0]
    rng=np.random.default_rng(468)
    people_draw=rng.integers(9,size=(10000,9,1,1))
    day_draw=rng.integers(2,size=(10000,9,2,1))
    order_draw=rng.integers(2,size=(10000,9,2,2))
    interval=lambda x: np.quantile(x,[.1,.9]).tolist()
    pooled=interval(gap[people_draw,day_draw,order_draw].mean((1,2,3)))
    per_order={}
    for j,order in enumerate(ORDERS):
        delta=gap[:,:,j]
        per_order[order]=dict(pp=float(delta.mean()),
             paired80=interval(delta[people_draw[...,0],day_draw[...,0]].mean((1,2))))
    correct=(pred[2]==y)
    prior=(pred[0]==y)
    result=dict(orders=ORDERS,absolute=dict(zip(ARMS,cells.mean((1,2,3)).tolist())),
        pp=float(gap.mean()),paired80=pooled,by_order=per_order,
        people=dict(zip(roster,gap.mean((1,2)).tolist())),
        advance=pooled[0]>0 and all(x['paired80'][0]>0 for x in per_order.values()),
        changed=int(np.count_nonzero(pred[2]!=pred[0])),
        corrections=int(np.count_nonzero(correct&~prior)),
        regressions=int(np.count_nonzero(prior&~correct)),stock_anchor_sha=sha(PRIOR/'result.json'),
        before_labels=sha(Path('before_labels.json')))
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':
    collect()
