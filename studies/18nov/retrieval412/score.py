"""Score actual frozen and JS endpoints, only after sealing every actor."""
import json
from pathlib import Path
import sys
import numpy as np
from actor import ARMS,PEOPLE,SEEDS,root,joint,lock

sys.path.insert(0,str(root/'ni26_fusion383_v1_20261118'))
from experiment import interval
from fusion_inputs import CHAIN,member_bank,official,probability,restore,sha


def main():
    assert Path('replay.log').read_text().splitlines()[-1]=='OK'
    assert lock()==json.loads(Path('fits.json').read_text())
    sc,scores,meta,ages=restore()
    assert sc.BASES[0]=='B2_pool'
    sealed={}
    pred={}
    active={}
    for p in PEOPLE:
        rows=np.flatnonzero(meta['person']==p)
        path=Path(p+'.npz')
        rec=json.loads(path.with_suffix('.json').read_text())
        assert rec==dict(source=sha(Path('source.sha256')),fits=sha(Path('fits.json')),outputs=sha(path),held=p,labels_opened=False)
        sealed[p]=rec
        bank=member_bank(382,p,rows,sc.ORDERS)
        with np.load(path) as z:
            assert np.array_equal(rows,z['index'])
            for o in sc.ORDERS:
                bases={'whole':z['native|'+o]}
                for branch,ref in [('control','frozen'),('JS','JS')]:
                    prob=probability(scores[branch,o,'B2_pool'][rows],ages[o][rows])
                    assert np.array_equal(prob.argmax(1),bank[ref,o].argmax(1))
                    bases[branch]=np.log(np.maximum(prob,np.finfo(float).tiny))
                for seed in SEEDS:
                    for kind,base in bases.items():
                        zero=base.argmax(1)
                        pred[seed,p,o,kind,'zero']=zero
                        for arm in ARMS:
                            delta=z[f'{seed}|{arm}|{o}']
                            assert np.isfinite(delta).all() and np.max(abs(delta.sum(1)))<1e-6
                            final=(base+delta).argmax(1)
                            pred[seed,p,o,kind,arm]=final
                            key=f'{seed}|{kind}|{arm}'
                            active[key]=active.get(key,0)+int((final!=zero).sum())
    Path('before_labels.json').write_text(json.dumps(dict(actors=sealed,decisions={
        '|'.join(map(str,k)):joint.digest(v) for k,v in pred.items()})))
    with np.load(CHAIN/'data/proteus_windows.npz') as d:
        y=d['y']
    metric=official()
    out={}
    for seed in SEEDS:
        for kind,expected in [('whole',60.78519278107362),('control',81.21561492579204),('JS',81.23552194696784)]:
            vals={}
            for arm in (*ARMS,'zero'):
                arr=np.empty((4,8))
                for i,p in enumerate(PEOPLE):
                    mask=meta['person']==p
                    for j,o in enumerate(sc.ORDERS):
                        res=metric(y[mask],pred[seed,p,o,kind,arm],meta,mask)
                        assert res['n_cells']==2
                        arr[i,j]=100*res['balanced_accuracy_cell_mean']
                vals[arm]=arr
            assert abs(interval(vals['zero'])['pp']-expected)<1e-8
            for a in ('zero','destroyed','cyclic','donor'):
                vals['real-'+a]=vals['real']-vals[a]
            out[f'{seed}|{kind}']={a:interval(v) for a,v in vals.items()}
    Path('result.json').write_text(json.dumps(dict(scores=out,changed=active,proof=sha(Path('before_labels.json'))),indent=2))


if __name__=='__main__':
    main()
