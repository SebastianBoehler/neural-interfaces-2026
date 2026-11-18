"""Authenticate balanced and stock releases, then calculate the paired metric."""
import json
from pathlib import Path

import numpy as np

from data import permutation,sha
from protocol import score

ORDERS=(411,1411)
ARMS=('stock','primary')
PROBE=Path('../neural-interfaces-2026/data/scherer2015/windows.npz')
PRIOR=Path('../ni26_cross411b_20261118')


def restore(people,days,groups):
    roster=sorted(set(people))
    perms={seed:permutation(groups,None,'full',seed) for seed in ORDERS}
    pred=np.full((2,2,len(people)),-1)
    receipts,mass={},[]
    for i,person in enumerate(roster):
        prior=json.loads((PRIOR/f'cap_{i}.json').read_text())
        for j,arm in enumerate(ARMS):
            p=Path(f'{arm}_{i}.json')
            h=sha(p)
            assert h==p.with_suffix('.sha256').read_text().strip()
            rec=json.loads(p.read_text())
            assert rec['source']==sha(Path('source.sha256'))
            assert rec['sessions'].keys()==prior.keys() and len(prior)==4
            for key,v in rec['sessions'].items():
                who,day,seed=key.split('|')
                seed=int(seed)
                rows=perms[seed][(people==person)&(days==day)]
                k=ORDERS.index(seed)
                old=prior[key]
                assert who==person and rows.tolist()==v['rows']==old['rows']
                assert v['activity']==old['activity'] and v['labels'][:20]==old['labels'][:20]
                if arm=='stock':
                    assert v['labels']==old['labels']
                for e in v['events']:
                    a,b=(np.asarray(e['coefficient_mass_'+phase]) for phase in ('before','after'))
                    shift=b[a.argmax()]-a.max()
                    assert e['unequal'] and (shift==0 if arm=='stock' else shift<0)
                    if arm=='primary':
                        mass.append(float(shift))
                assert (pred[j,k,rows]==-1).all()
                pred[j,k,rows]=v['labels']
            receipts[p.name]=h
    assert np.isin(pred,(0,1,2)).all()
    Path('before_labels.json').write_text(json.dumps(receipts,indent=2)+'\n')
    return roster,pred,mass


def collect():
    with np.load(PROBE,allow_pickle=False) as z:
        people,days,interface=(z[k] for k in ('subject','session','interface'))
    groups=np.char.add(np.char.add(people,'|'),days)
    roster,pred,mass=restore(people,days,groups)
    with np.load(PROBE,allow_pickle=False) as z:
        y=z['y']
    cells=np.empty((2,9,2,2))
    for a in range(2):
        for i,person in enumerate(roster):
            for d,day in enumerate(sorted(set(days[people==person]))):
                m=(people==person)&(days==day)
                assert np.unique(y[m]).tolist()==[0,1,2]
                for o in range(2):
                    cells[a,i,d,o]=100*np.mean([(pred[a,o,m&(y==c)]==c).mean() for c in range(3)])
        for j in range(2):
            official=score(y,pred[a,j],dict(stream=groups,interface=interface),np.ones(len(y),bool))
            assert official['n_cells']==18
            np.testing.assert_allclose(100*official['balanced_accuracy_cell_mean'],
                                       cells[a,:,:,j].mean(),atol=1e-10,rtol=0)
    assert abs(cells[0].mean()-json.loads((PRIOR/'result.json').read_text())['acc'][1])<1e-10
    delta=cells[1]-cells[0]
    rng=np.random.default_rng(4694)
    samples=(rng.integers(9,size=(10000,9,1,1)),rng.integers(2,size=(10000,9,2,1)),
             rng.integers(2,size=(10000,9,2,2)))
    ci=np.quantile(delta[samples].mean((1,2,3)),[.1,.9]).tolist()
    by_order=dict(zip(ORDERS,delta.mean((0,1)).tolist()))
    stock,balanced=np.equal(pred,y)
    supported=ci[0]>0 and min(by_order.values())>0
    result=dict(absolute=dict(zip(('stock','balanced'),cells.mean((1,2,3)).tolist())),
        pp=float(delta.mean()),paired80=ci,by_order=by_order,
        by_person=dict(zip(roster,delta.mean((1,2)).tolist())),
        changed=int(np.count_nonzero(pred[0]!=pred[1])),corrections=int(np.count_nonzero(balanced&~stock)),
        regressions=int(np.count_nonzero(stock&~balanced)),decision_receipt=sha(Path('before_labels.json')),
        source=sha(Path('source.sha256')),balanced_refits=len(mass),
        mean_dominant_mass_change_pp=100*float(np.mean(mass)),supported=supported,
        meets_accuracy_advance=supported and delta.mean()>=.30)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':
    collect()
