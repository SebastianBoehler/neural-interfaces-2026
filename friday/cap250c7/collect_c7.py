"""Seal all paired worker decisions, then score on the independent cohort."""
import json
from pathlib import Path

import numpy as np
from data import permutation,sha
from protocol import score

ARMS=('stock','inner','causal','permuted','learned')
EXTERNAL=Path('../neural-interfaces-2026/data/scherer2015/windows.npz')
ARCHIVE=Path('../ni26_qualify385_v1_20261118')


def collect():
    with np.load(EXTERNAL,allow_pickle=False) as z:
        people,days,interface=(z[k] for k in ('subject','session','interface'))
    roster=sorted(set(people))
    sessions=np.char.add(np.char.add(people,'|'),days)
    orders=[permutation(sessions,None,'full',s) for s in (0,1)]
    coef_sha=sha(Path('coefficients.json'))
    source_sha=sha(Path('source.sha256'))
    predictions=np.full((len(ARMS),2,len(people)),-1)
    old_mismatch=0
    receipts={}
    for i,person in enumerate(roster):
        path=Path(f'held_{i}.json')
        assert sha(path)==Path(str(path)+'.sha256').read_text().strip()
        lock=json.loads(path.read_text())
        assert lock['coefficient_sha']==coef_sha and lock['source']==source_sha
        prior=json.loads((ARCHIVE/f'cap_{i}.json').read_text())
        assert lock['sessions'].keys()==prior.keys() and len(prior)==4
        for key,rec in lock['sessions'].items():
            owner,day,order=key.split('|')
            order=int(order)
            rows=orders[order][(people==person)&(days==day)]
            original=prior[key]
            assert owner==person and rows.tolist()==rec['rows']==original['rows']
            assert original['activity']==rec['activity']
            labels=rec['predictions']
            assert labels['stock']==original['labels']
            assert set(labels)==set(ARMS) and (predictions[:,order,rows]==-1).all()
            for a,arm in enumerate(ARMS):
                predictions[a,order,rows]=labels[arm]
            disagree=np.asarray(labels['stock'])!=labels['inner']
            for arm in ('causal','permuted','learned'):
                assert np.array_equal(np.asarray(labels[arm])[~disagree],np.asarray(labels['stock'])[~disagree])
            old_mismatch+=rec['standalone_old_label_differences']
        receipts[path.name]=sha(path)
    assert np.isin(predictions,(0,1,2)).all()
    Path('before_labels.json').write_text(json.dumps(receipts,indent=2)+'\n')

    with np.load(EXTERNAL,allow_pickle=False) as z:
        y=z['y']
    cells=np.empty((len(ARMS),len(roster),2,2))
    for a,arm in enumerate(ARMS):
        for i,person in enumerate(roster):
            for d,day in enumerate(sorted(set(days[people==person]))):
                mask=(people==person)&(days==day)
                assert np.unique(y[mask]).tolist()==[0,1,2]
                for new in (0,1):
                    cells[a,i,d,new]=100*np.mean([
                        (predictions[a,new,mask&(y==c)]==c).mean() for c in range(3)])
        for j in (0,1):
            official=score(y,predictions[a,j],dict(stream=sessions,interface=interface),
                           np.ones(len(y),bool))
            assert official['n_cells']==18
            np.testing.assert_allclose(100*official['balanced_accuracy_cell_mean'],
                                       cells[a,:,:,j].mean(),atol=1e-10,rtol=0)
    baseline=json.loads((ARCHIVE/'result.json').read_text())['acc'][1]
    assert abs(cells[0].mean()-baseline)<1e-9
    rnd=np.random.default_rng(467)
    draw=tuple(rnd.integers(n,size=s) for n,s in (
        (9,(10000,9,1,1)),(2,(10000,9,2,1)),(2,(10000,9,2,2))))
    contrasts={}
    for left,right in (('learned','stock'),('learned','permuted'),('learned','causal'),
                       ('inner','stock'),('causal','stock'),('permuted','stock')):
        a,b=ARMS.index(left),ARMS.index(right)
        delta=cells[a]-cells[b]
        v=predictions[a]==y
        t=predictions[b]==y
        contrasts[left+'-'+right]=dict(pp=float(delta.mean()),
            paired80=np.quantile(delta[draw].mean((1,2,3)),[.1,.9]).tolist(),
            per_person=dict(zip(roster,delta.mean((1,2)).tolist())),
            per_order=delta.mean((0,1)).tolist(),
            changes=int(np.count_nonzero(predictions[a]!=predictions[b])),
            corrections=int(np.count_nonzero(v&~t)),regressions=int(np.count_nonzero(t&~v)))
    result=dict(absolute=dict(zip(ARMS,cells.mean((1,2,3)).tolist())),contrasts=contrasts,
                standalone_old_differences=old_mismatch,coefficient_sha=coef_sha,
                before_labels=sha(Path('before_labels.json')))
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['absolute'],{k:v['pp'] for k,v in contrasts.items()},flush=True)


if __name__=='__main__':
    collect()
