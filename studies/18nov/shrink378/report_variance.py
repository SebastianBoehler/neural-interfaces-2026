"""Paired held-person/order estimates for future variance prediction accuracy."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ARMS=('K','G','P','RV')
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
AGES=(1,8,32,90)
PEOPLE=tuple(f'P{i:02d}' for i in range(1,11))
SOURCE='9931367d9b911eb1377c3a91c26066f076e6e8d0a9be3dba69b172b45958ea01'


def matrix(root):
    # person × order × age × arm × (MSE, MAE), equal seeds and sessions
    values=np.empty((10,8,4,4,2))
    provenance={}
    for pi,p in enumerate(PEOPLE):
        raw=(root/(p+'.json')).read_bytes()
        record=json.loads(raw)
        assert record['person']==p and record['held_labels_used'] is False and record['source_sha']==SOURCE
        provenance[p]=hashlib.sha256(raw).hexdigest()
        groups={}
        for seed,o,s,age,arm,mse,mae in record['rows']:
            key=(o,age,arm)
            assert (seed,s) not in groups.setdefault(key,{})
            groups[key][seed,s]=(mse,mae)
        for oi,o in enumerate(ORDERS):
            for ai,age in enumerate(AGES):
                for mi,arm in enumerate(ARMS):
                    group=groups[o,age,arm]
                    assert len(group)==(6 if p in ('P04','P10') else 2)
                    values[pi,oi,ai,mi]=np.mean(list(group.values()),0)
    assert np.isfinite(values).all() and np.all(values>=0)
    return values,provenance


def main(root,destination):
    v,provenance=matrix(root)
    rng=np.random.default_rng(378)
    scheme=[[0,1,2,3],[4,5],[6,7]]
    central=np.mean([v[:,g].mean(1) for g in scheme],axis=0)
    # Sample people jointly for all arms, then draw their paired orders by scheme.
    people=rng.integers(10,size=(4000,10))
    boot=np.zeros((4000,4))
    for g in scheme:
        orders=rng.choice(g,size=(4000,10,len(g)))
        # Average age first to bound memory.
        age_avg=v[...,0].mean(2)
        boot+=age_avg[people[:,:,None],orders].mean((1,2))/3
    obs=central[...,0].mean((0,1))
    contrasts={}
    for i,ref in enumerate(ARMS[1:],1):
        point=100*(1-obs[0]/obs[i])
        ci=100*(1-boot[:,0]/boot[:,i])
        by_person=central[...,0].mean(1)
        contrasts['K_vs_'+ref]=dict(mse_reduction_pct=point,paired80=np.percentile(ci,[10,90]).tolist(),
            by_person_pct=dict(zip(PEOPLE,(100*(1-by_person[:,0]/by_person[:,i])).tolist())))
    result=dict(loss='mean squared log variance ratio; lower is better',
                mse_by_arm=dict(zip(ARMS,obs.tolist())),contrasts=contrasts,
                age_mae={age:dict(zip(ARMS,central[:,j,:,-1].mean(0).tolist())) for j,age in enumerate(AGES)},
                folds=provenance)
    destination.write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]))
