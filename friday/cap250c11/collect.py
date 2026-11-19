"""Authenticate exact workers, then score predictions and earlier target edits."""
import json
from pathlib import Path

import numpy as np
from data import permutation,sha
from protocol import score
from verify import ARMS,events,load

ORDERS=(0,1)
EXTERNAL=Path('../neural-interfaces-2026/data/scherer2015/windows.npz')
PRIOR=Path('../ni26_qualify385_v1_20261118')


def align(people,days,groups):
    roster=sorted(set(people))
    orders=[permutation(groups,None,'full',o) for o in ORDERS]
    full=np.full((3,2,len(people)),-1)
    receipts,tracks={},{}
    n_hard,moved=0,0
    for i,person in enumerate(roster):
        bank,hashes=load(i,False)
        receipts.update(hashes)
        cap,old=(json.loads((PRIOR/f'{a}_{i}.json').read_text()) for a in ('cap','old'))
        assert bank['stock'].keys()==cap.keys()==old.keys() and len(cap)==4
        for key in cap:
            who,day,order=key.split('|')
            o=int(order)
            rows=orders[o][(people==person)&(days==day)]
            assert who==person and rows.tolist()==cap[key]['rows']==old[key]['rows']
            observations=[bank[arm][key] for arm in ARMS]
            s=observations[0]
            assert s['labels']==cap[key]['labels'] and s['activity']==cap[key]['activity']
            assert s['inner']==old[key]['labels']
            for j,item in enumerate(observations):
                assert item['rows']==rows.tolist() and (full[j,o,rows]==-1).all()
                full[j,o,rows]=item['labels']
            for triplet in zip(*(x['events'] for x in observations)):
                moved+=events(*triplet)
                n_hard+=triplet[0]['age']//2
            tracks[key]=(rows,observations)
    assert np.isin(full,(0,1,2)).all()
    Path('before_labels.json').write_text(json.dumps(receipts,indent=2)+'\n')
    return roster,full,tracks,n_hard,moved


def metrics(people,days,interface,y,groups,roster,pred):
    cells=np.empty((3,len(roster),2,2))
    for a in range(3):
        for i,person in enumerate(roster):
            for j,day in enumerate(sorted(set(days[people==person]))):
                mask=(people==person)&(days==day)
                assert np.unique(y[mask]).tolist()==[0,1,2]
                for o in range(2):
                    cells[a,i,j,o]=100*np.mean([(pred[a,o,mask&(y==c)]==c).mean() for c in range(3)])
        for o in range(2):
            expected=score(y,pred[a,o],dict(stream=groups,interface=interface),np.ones(len(y),bool))
            assert expected['n_cells']==18
            np.testing.assert_allclose(100*expected['balanced_accuracy_cell_mean'],
                                        cells[a,:,:,o].mean(),atol=1e-10,rtol=0)
    stock=json.loads((PRIOR/'result.json').read_text())['acc'][1]
    assert abs(stock-cells[0].mean())<1e-10
    return cells


def contrast(cells,pred,y,people):
    rng=np.random.default_rng(4611)
    draws=(rng.integers(9,size=(10000,9,1,1)),rng.integers(2,size=(10000,9,2,1)),
           rng.integers(2,size=(10000,9,2,2)))
    out={}
    for a,b in ((1,0),(1,2),(2,0)):
        diff=cells[a]-cells[b]
        left,right=(pred[i]==y for i in (a,b))
        out[f'{ARMS[a]}-{ARMS[b]}']=dict(pp=float(diff.mean()),
             paired80=np.quantile(diff[draws].mean((1,2,3)),[.1,.9]).tolist(),
             by_order=diff.mean((0,1)).tolist(),by_person=dict(zip(people,diff.mean((1,2)).tolist())),
             changed=int(np.count_nonzero(pred[a]!=pred[b])),
             corrections=int(np.count_nonzero(left&~right)),regressions=int(np.count_nonzero(right&~left)))
    return out


def target_diagnostics(tracks,y,n_hard,moved):
    n=np.zeros((2,3),int)
    times=[]
    for key,(rows,arms) in tracks.items():
        if not key.endswith('|1|0'):
            times.append([v['seconds']/arms[0]['seconds'] for v in arms[1:]])
        for a,arm in enumerate(arms[1:]):
            for e in arm['events']:
                for i,label in e['edited']:
                    truth=y[rows[i]]
                    n[a]+=(1,truth==e['raw'][i],truth==label)
    assert n[0,0]==n[1,0] and len(times)==27
    return dict(hard_exposures=n_hard,distinct_control_refits=moved,
        edited_exposures=int(n[0,0]),edit_fraction=float(n[0,0]/n_hard),
        correct_before=n[:,1].tolist(),correct_after=n[:,2].tolist(),
        median_warm_time_ratio=np.median(times,axis=0).tolist())


def main():
    with np.load(EXTERNAL,allow_pickle=False) as z:
        people,days,interface=(z[k] for k in ('subject','session','interface'))
    groups=np.char.add(np.char.add(people,'|'),days)
    roster,pred,tracks,n_hard,moved=align(people,days,groups)
    with np.load(EXTERNAL,allow_pickle=False) as z:
        y=z['y']
    cells=metrics(people,days,interface,y,groups,roster,pred)
    c=contrast(cells,pred,y,roster)
    gain=c['geometry-stock']
    support=(gain['paired80'][0]>0 and min(gain['by_order'])>0 and
             c['geometry-rotation']['paired80'][0]>0)
    result=dict(absolute=dict(zip(ARMS,cells.mean((1,2,3)).tolist())),contrasts=c,
        training_targets=target_diagnostics(tracks,y,n_hard,moved),advance=support,
        source=sha(Path('source.sha256')),decision_receipt=sha(Path('before_labels.json')))
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':
    main()
