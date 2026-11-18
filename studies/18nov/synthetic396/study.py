"""Recombine measured source-session spectral offsets in coherent or shuffled episodes."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

root = Path('/mnt/lustre/work/gehler/gfh527')
dyn = root / 'ni26_dynamics389_v1_20261118'
fusion = root / 'ni26_fusion383_v1_20261118'
sys.path[:0] = [str(dyn), str(fusion)]
from common import DATA, ORDERS, PEOPLE, metadata, orders, sha
from fusion_inputs import BRANCHES, official, probability, restore
from score389 import summary

def features(x, prior):
    n = np.arange(1, len(x)+1)[:, None]
    center = (20*prior + np.cumsum(x, axis=0))/(20+n)
    return np.column_stack((x, x-center))

def load():
    assert sha(dyn/'source.sha256')=='fc284f7948aa6f4e43fb9d4299f452b3cb0c5330d0602e80f050239901adea97'
    record=json.loads((dyn/'inputs.json').read_text())
    gram=np.load(dyn/'gram.npy',mmap_mode='r')
    assert record['gram'] == sha(dyn/'gram.npy') == '0e8036e215c575364c94552c480213dab86707971f9320b2136138c3a836e561'
    # Last 12 channels store the current samples, with two time halves.
    power=gram[...,48:,48:].diagonal(axis1=-2,axis2=-1).mean(axis=1)
    assert np.isfinite(power).all() and power.min()>0
    return np.log(power.reshape(len(power),-1).astype(float))

def train(x, y, streams):
    rng=np.random.default_rng(396)
    sessions=[np.flatnonzero(streams==s) for s in np.unique(streams)]
    style=np.stack([np.mean([x[rows[y[rows]==c]].mean(0) for c in range(3)],axis=0)
                    for rows in sessions])
    prior=style.mean(0)
    banks={arm:[] for arm in ('duplicate','coherent','independent')}
    labels=[]
    for epoch in range(4):
        shuffled=[rng.permutation(rr) for rr in sessions]
        shifts=[]
        for rr in shuffled:
            a,b=rng.choice(len(sessions),2,replace=False)
            shifts.append(np.broadcast_to(.5*(style[a]-style[b]),(len(rr),x.shape[1])))
        joined=np.concatenate(shifts)
        independent=np.split(rng.permutation(joined),np.cumsum([len(r) for r in shuffled])[:-1])
        np.testing.assert_allclose(np.concatenate(independent).mean(0),joined.mean(0),atol=1e-10)
        for rr,offset,mixed in zip(shuffled,shifts,independent):
            clean=features(x[rr],prior)
            for arm,new in [('duplicate',clean),('coherent',features(x[rr]+offset,prior)),
                            ('independent',features(x[rr]+mixed,prior))]:
                banks[arm].extend([clean,new])
            labels.extend([y[rr],y[rr]])
    target=np.concatenate(labels)
    norm=StandardScaler().fit(np.concatenate(banks['duplicate']))
    models={arm:LogisticRegression(C=.01,max_iter=300,class_weight='balanced').fit(
        norm.transform(np.concatenate(bank)),target) for arm,bank in banks.items()}
    return models,norm,prior

def fold(i,smoke):
    meta=metadata()
    x=load()
    rows=np.flatnonzero(meta['person']==PEOPLE[i])
    src=np.flatnonzero(meta['person']!=PEOPLE[i])
    with np.load(DATA) as data:
        y=data['y'][src]
    models,norm,prior=train(x[src],y,meta['stream'][src])
    result={'rows':rows}
    for order,perm in orders().items():
        for arm in models:
            result[arm+'|'+order]=np.zeros((len(rows),3))
        for session in np.unique(meta['stream'][rows]):
            seq=perm[meta['stream'][perm]==session]
            assert np.isin(seq,rows).all()
            feat=features(x[seq],prior)
            np.testing.assert_array_equal(feat[:7],features(x[seq[:7]],prior))
            for arm,m in models.items():
                result[arm+'|'+order][np.searchsorted(rows,seq)]=m.predict_proba(norm.transform(feat))
    assert all(np.allclose(v.sum(1),1) for k,v in result.items() if k!='rows')
    if not smoke:
        np.savez_compressed(PEOPLE[i]+'.npz',**result)

def score():
    sc,banks,meta,age=restore()
    assert tuple(sc.ORDERS)==ORDERS
    metric=official()
    tables={}
    counts=np.empty(10,int)
    paths=[Path(p+'.npz') for p in PEOPLE]
    Path('before_labels.json').write_text(json.dumps({str(p):sha(p) for p in paths}))
    with np.load(DATA) as data:
        y=data['y']
    for i,person in enumerate(PEOPLE):
        mask=meta['person']==person
        with np.load(paths[i]) as z:
            assert np.array_equal(z['rows'],np.flatnonzero(mask))
            for j,o in enumerate(ORDERS):
                model={a:z[a+'|'+o] for a in ('duplicate','coherent','independent')}
                preds=dict(model)
                for b in BRANCHES:
                    q=probability(banks[b,o,'B2_pool'][mask],age[o][mask])
                    preds[b]=q
                    preds.update({b+'_'+a:.95*q+.05*v for a,v in model.items()})
                for a,v in preds.items():
                    r=metric(y[mask],v.argmax(1),meta,mask)
                    counts[i]=r['n_cells']
                    tables.setdefault(a,np.empty((8,10)))[j,i]=100*r['balanced_accuracy_cell_mean']
    vals={a:summary(v,counts)['pp'] for a,v in tables.items()}
    assert abs(vals['control']-76.27933502145761)<1e-7 and abs(vals['JS']-76.55173151211109)<1e-7
    diffs={}
    for prefix in ('','control_','JS_'):
        for ref in ('duplicate','independent'):
            diffs[prefix+'coherent-'+ref]=summary(tables[prefix+'coherent']-tables[prefix+ref],counts)
    for b in BRANCHES:
        diffs[b+'_coherent-stock']=summary(tables[b+'_coherent']-tables[b],counts)
    Path('result.json').write_text(json.dumps(dict(scores=vals,contrasts=diffs),indent=2))

if __name__=='__main__':
    if sys.argv[1]=='score':
        score()
    else:
        fold(int(sys.argv[2]),sys.argv[1]=='smoke')
