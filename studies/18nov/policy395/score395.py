"""Commit all calibrated streams, then rescore official native seeds and complete systems."""
import json
from pathlib import Path
import sys
import numpy as np
from policy import ARMS
from substrate import DATA,DYN,OLD,ORDERS,PEOPLE,WORK,digest,metadata,sha
sys.path.insert(0,str(DYN))
from score389 import controls_and_age,summary,source_manifest,METRICS


def main():
    meta=metadata(); base,_=controls_and_age(meta)
    out={f'{s}|{a}':np.empty((8,2655),np.int8) for s in (0,1,'ensemble') for a in ARMS}
    proofs={}
    for p in PEOPLE:
        rr=np.flatnonzero(meta['person']==p); fit=np.flatnonzero(meta['person']!=p)
        rec=json.loads(Path(p+'.json').read_text())
        assert rec['held']==p and rec['fit']==digest(fit) and rec['target']==digest(rr)
        assert rec['held_labels_opened'] is False and rec['source']==sha('source.sha256') and rec['outputs']==sha(p+'.npz')
        assert rec['prefix']<1e-10 and rec['one_step']<1e-10
        proofs[p]=rec
        with np.load(p+'.npz') as z:
            assert np.array_equal(z['index'],rr)
            for i,o in enumerate(ORDERS):
                for a in ARMS:
                    probs=[z[f'{s}|{a}|{o}'] for s in (0,1)]
                    for s,v in enumerate(probs):
                        assert v.shape==(len(rr),3) and np.isfinite(v).all() and np.allclose(v.sum(1),1)
                        out[f'{s}|{a}'][i,rr]=v.argmax(1)
                    out['ensemble|'+a][i,rr]=np.mean(probs,0).argmax(1)
    before=Path('before_labels.json')
    before.write_text(json.dumps(dict(outputs={p:d['outputs'] for p,d in proofs.items()},source=sha('source.sha256'),
                                     held_labels_opened=False),indent=2))
    assert source_manifest(METRICS/'source.sha256')=='9e889431c92f4759c79e0724446d19fc9503ac5b726464f73f9ed22b548f9ccd'
    sys.path.insert(0,str(METRICS))
    from measure import official
    metric=official()
    with np.load(DATA) as z:
        gold=z['y']
    tables={}
    counts=np.zeros(10,int)
    for a,v in {**base,**out}.items():
        table=np.empty((8,10))
        for i,o in enumerate(ORDERS):
            for j,p in enumerate(PEOPLE):
                mask=meta['person']==p; result=metric(gold[mask],v[i,mask],meta,mask)
                table[i,j]=100*result['balanced_accuracy_cell_mean']; counts[j]=result['n_cells']
        tables[a]=table
    scores={k:summary(v,counts)['pp'] for k,v in tables.items()}
    assert abs(scores['JS']-76.55173151211109)<1e-7 and abs(scores['frozen']-76.27933502145761)<1e-7
    contrasts={}
    for seed in ('0','1','ensemble'):
        key=seed+'|learned'
        for b in ('native','fixed','mean','donor','static','frozen','JS'):
            rhs=b if b in ('frozen','JS') else seed+'|'+b
            contrasts[key+' minus '+rhs]=summary(tables[key]-tables[rhs],counts)
        contrasts[seed+'|fixed minus '+seed+'|native']=summary(tables[seed+'|fixed']-tables[seed+'|native'],counts)
    Path('result.json').write_text(json.dumps(dict(before=sha(before),scores=scores,contrasts=contrasts),indent=2))
    print(scores,flush=True)


if __name__=='__main__':
    main()
