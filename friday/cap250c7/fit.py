"""Prepare and fit the two policies using only the authenticated public source."""
import json
from pathlib import Path
import sys

import numpy as np
import score379 as ref
from arbitrate import log_scores
from produce import new_stock,public_inputs
from produce_c6 import archived_old,LINEAGE,LINEAGE_SHA

from features import batch,CausalFeatures,PAIRS,select


def source(smoke):
    assert ref.sha(LINEAGE)==LINEAGE_SHA
    pins = json.loads(LINEAGE.read_text())['sha256']
    meta,permutations = public_inputs()
    records = []
    proof = {}
    for order in ref.ORDERS[:1] if smoke else ref.ORDERS:
        old, digest = archived_old(order,False,pins)
        _,_,cap,stock = new_stock(order)
        with np.load(permutations,allow_pickle=False) as z:
            delivery = z[f'{order[:-1]}|{order[-1]}|all_loso']
        assert np.array_equal(np.sort(delivery),np.arange(len(old)))
        for stream in sorted(set(meta['stream'])):
            rows = delivery[meta['stream'][delivery]==stream]
            age = np.arange(len(rows))
            a,b = log_scores(old[rows],age>=0),log_scores(cap[rows],age<90)
            x,old_label,cap_label = batch(a,b)
            assert np.array_equal(old_label,old[rows].argmax(1))
            assert np.array_equal(cap_label,cap[rows].argmax(1))
            if smoke:
                state=CausalFeatures()
                stepwise=np.stack([state.step(u,v)[0] for u,v in zip(a,b)])
                np.testing.assert_allclose(stepwise,x,atol=1e-12,rtol=0)
            take=old_label!=cap_label
            scheme_weight=1/(12 if order.startswith('within') else 6)
            records.append((rows[take],x[take],old_label[take],cap_label[take],scheme_weight))
        proof[order]=dict(old=digest,stock=stock)
    return records,proof,meta


def ridge(z,y,w):
    mean = (w[:,None]*z).sum(0)
    scale = np.maximum(.05,np.sqrt((w[:,None]*(z-mean)**2).sum(0)))
    x = (z-mean)/scale
    bias = w@y
    beta = np.linalg.solve(x.T@(w[:,None]*x)+.10*np.eye(x.shape[1]),x.T@(w*(y-bias)))
    coef = dict(mean=mean.tolist(),scale=scale.tolist(),beta=beta.tolist(),bias=float(bias))
    assert np.isfinite(beta).all() and np.linalg.norm(beta)>0
    return coef


def fit(smoke):
    records,proof,meta = source(smoke)
    with np.load(ref.BASE/'data/proteus_windows.npz',allow_pickle=False) as z:
        y = z['y']
    assert set(meta['person'])==set(ref.PEOPLE)
    keys = np.char.add(np.char.add(meta['stream'],'|'),meta['interface'])
    keys = np.char.add(np.char.add(keys,'|'),y.astype(str))
    _,mapping,counts = np.unique(keys,return_inverse=True,return_counts=True)
    weight = 1/counts[mapping]
    x,a,b,t,w = [],[],[],[],[]
    for rows,features,old,cap,order_weight in records:
        x.append(features)
        a.append(old)
        b.append(cap)
        truth = y[rows]
        t.append((old==truth).astype(float)-(cap==truth))
        w.append(weight[rows]*order_weight)
    x,a,b,t,w = map(np.concatenate,(x,a,b,t,w))
    w /= w.sum()
    assert set(np.unique(t))=={-1.,0.,1.} and len(t)>100
    null = t.copy()
    rnd = np.random.default_rng(467)
    pair = 3*a+b
    for code in PAIRS:
        idx = np.flatnonzero(pair==code)
        null[idx] = rnd.permutation(t[idx])
        assert np.array_equal(np.sort(null[idx]),np.sort(t[idx]))
    assert np.any(null!=t)
    coefficients = {k:ridge(x,v,w) for k,v in (('learned',t),('permuted',null))}
    selections = {k:int(np.count_nonzero(select(x,a,b,p)!=b)) for k,p in coefficients.items()}
    print('source disagreements',len(t),'switch counts',selections,flush=True)
    if not smoke:
        Path('coefficients.json').write_text(json.dumps(dict(coefficients=coefficients,
            source=ref.sha(Path('source.sha256')),source_input=ref.DATA_SHA,orders=ref.ORDER_SHA,
            input_parents=proof,source_people=list(ref.PEOPLE),disagreements=len(t),
            source_switch_counts=selections),indent=2)+'\n')


if __name__ == '__main__':
    fit(sys.argv[1:] == ['smoke'])
