"""Check F-C9 external decisions and score whole exact streamed systems."""
import json
from pathlib import Path

import numpy as np

from data import permutation, sha
from protocol import score

PRIOR = Path('../ni26_cross411b_20261118')
EXTERNAL = Path('../neural-interfaces-2026/data/scherer2015/windows.npz')
ORDERS = (411, 1411)
ARMS = ('stock','fixed','adaptive')


def restore(people, days, groups):
    roster = sorted(set(people))
    permutations={order:permutation(groups,None,'full',order) for order in ORDERS}
    predictions=np.full((3,2,len(people)),-1)
    receipts, scales, times={}, [], {}
    corrections={'fixed':0,'adaptive':0}
    for i,person in enumerate(roster):
        path=Path(f'held_{i}.json')
        assert sha(path)==path.with_suffix('.sha256').read_text().strip()
        record=json.loads(path.read_text())
        assert record['source']==sha(Path('source.sha256'))
        cap,old=[json.loads((PRIOR/f'{arm}_{i}.json').read_text()) for arm in ('cap','old')]
        assert record['sessions'].keys()==cap.keys()==old.keys() and len(cap)==4
        for name,session in record['sessions'].items():
            owner,day,order=name.split('|')
            order=int(order)
            rows=permutations[order][(people==person)&(days==day)]
            assert owner==person and rows.tolist()==session['rows']==cap[name]['rows']==old[name]['rows']
            j=ORDERS.index(order)
            arms=session['arms']
            assert set(arms)==set(ARMS) and (predictions[:,j,rows]==-1).all()
            assert arms['stock']['labels']==cap[name]['labels']
            for a,arm in enumerate(ARMS):
                value=arms[arm]
                assert value['inner']==old[name]['labels'] and value['activity']==cap[name]['activity']
                predictions[a,j,rows]=value['labels']
                if arm!='stock':
                    corrections[arm]+=value['changed']
            scales += [scale for trace in arms['adaptive']['coefficient'].values() for _,scale in trace]
            times[name]={arm:arms[arm]['wall_s'] for arm in ARMS}
        receipts[path.name]=sha(path)
    assert np.isin(predictions,(0,1,2)).all()
    Path('before_labels.json').write_text(json.dumps(receipts,indent=2)+'\n')
    return roster,predictions,scales,times,corrections


def collect():
    with np.load(EXTERNAL,allow_pickle=False) as z:
        people,days,interface=(z[k] for k in ('subject','session','interface'))
    groups=np.char.add(np.char.add(people,'|'),days)
    roster,pred,scales,times,updates=restore(people,days,groups)
    with np.load(EXTERNAL,allow_pickle=False) as z:
        y=z['y']
    cells=np.empty((len(ARMS),len(roster),2,2))
    for a,arm in enumerate(ARMS):
        for i,person in enumerate(roster):
            for d,day in enumerate(sorted(set(days[people==person]))):
                mask=(people==person)&(days==day)
                assert np.unique(y[mask]).tolist()==[0,1,2]
                for j in range(2):
                    cells[a,i,d,j]=100*np.mean([(pred[a,j,mask&(y==c)]==c).mean() for c in range(3)])
        for j in range(2):
            official=score(y,pred[a,j],dict(stream=groups,interface=interface),np.ones(len(y),bool))
            assert official['n_cells']==18
            np.testing.assert_allclose(100*official['balanced_accuracy_cell_mean'],
                                       cells[a,:,:,j].mean(),atol=1e-10,rtol=0)
    anchor=json.loads((PRIOR/'result.json').read_text())['acc'][1]
    assert abs(cells[0].mean()-anchor)<1e-10
    rng=np.random.default_rng(4690)
    people_draw=rng.integers(9,size=(10000,9,1,1))
    day_draw=rng.integers(2,size=(10000,9,2,1))
    order_draw=rng.integers(2,size=(10000,9,2,2))
    interval=lambda s: np.quantile(s,[.1,.9]).tolist()
    contrasts={}
    for new,comparator in (('adaptive','stock'),('fixed','stock'),('adaptive','fixed')):
        a,b=(ARMS.index(k) for k in (new,comparator))
        delta=cells[a]-cells[b]
        p,q=(pred[i]==y for i in (a,b))
        contrasts[f'{new}-{comparator}']=dict(pp=float(delta.mean()),
              paired80=interval(delta[people_draw,day_draw,order_draw].mean((1,2,3))),
              by_order={s:float(delta[:,:,j].mean()) for j,s in enumerate(ORDERS)},
              by_person=dict(zip(roster,delta.mean((1,2)).tolist())),
              changed=int(np.count_nonzero(pred[a]!=pred[b])),
              corrections=int(np.count_nonzero(p&~q)),regressions=int(np.count_nonzero(q&~p)))
    measured=contrasts['adaptive-stock']
    advance=(measured['paired80'][0]>0 and min(measured['by_order'].values())>0
             and contrasts['adaptive-fixed']['paired80'][0]>0)
    result=dict(absolute=dict(zip(ARMS,cells.mean((1,2,3)).tolist())),contrasts=contrasts,
         coefficient=dict(n=len(scales),quantiles=np.quantile(scales,[0,.1,.5,.9,1]).tolist(),
                          strictly_edited=updates),
         worker_seconds=times,source=sha(Path('source.sha256')),
         before_labels=sha(Path('before_labels.json')),advance=advance)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['absolute'],{a:v['pp'] for a,v in contrasts.items()},advance,flush=True)


if __name__ == '__main__':
    collect()
