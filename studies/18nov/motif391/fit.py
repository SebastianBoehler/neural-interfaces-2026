"""Cross-fit task conditional categorical models before fitting the discriminative head."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import score379 as base
from source_labels import read_selected,read_training_labels

ARMS=('power','marginal','transition','scrambled','d1','d3','d8')


def likelihood(counts, templates):
    # shape N x 2 real/null x 36 views x 6 source symbols x 6 next symbols
    model=templates+.5
    log_trans=np.log(model/model.sum(-1,keepdims=True))
    mass=counts.sum((-1,-2))
    trans=np.einsum('nrvij,crvij->nrvc',counts,log_trans,optimize=True)/mass[...,None]
    occ=counts[:,0].sum(-2)
    marginal=model[:,0].sum(-2)
    log_occ=np.log(marginal/marginal.sum(-1,keepdims=True))
    ll=np.einsum('nvi,cvi->nvc',occ,log_occ,optimize=True)/occ.sum(-1)[...,None]
    return ll-ll.mean(-1,keepdims=True), trans-trans.mean(-1,keepdims=True)


def design(pow_,m,t,arm):
    result=[pow_]
    if arm!='power': result.append(m.reshape(len(m),-1))
    if arm not in ('power','marginal'):
        selected=t[:,int(arm=='scrambled')]
        if arm in ('d1','d3','d8'):
            selected=selected.reshape(len(m),3,12,3)[:,('d1','d3','d8').index(arm)]
        result.append(selected.reshape(len(m),-1))
    return np.column_stack(result)


def main(held,smoke):
    start=time.monotonic()
    path=base.BASE/'data/proteus_windows.npz'
    assert base.sha(path)==base.DATA_SHA
    perm=base.BASE/'runs/shuffle_perms.npz'
    assert base.sha(perm)==base.ORDER_SHA
    with np.load(path) as z:
        people=np.array([p.split('/')[-1] for p in z['subject']])
        days=z['session']
    train=np.flatnonzero(people!=held)
    test=np.flatnonzero(people==held)
    y=read_training_labels(path,train,len(people))
    interface=read_selected(path,'interface',train,len(people))
    cell=np.char.add(np.char.add(days[train],'|'),interface)
    file=Path('features.npz')
    with np.load(file) as z:
        assert np.array_equal(z['index'],np.arange(len(people)))
        power,counts=z['p'],z['c']
    source=sorted(set(people[train]))
    assert len(source)==9 and held not in source
    templates={}
    w=np.zeros(len(train))
    for p in source:
        mask=people[train]==p
        cells=sorted(set(cell[mask]))
        rows=[]
        for label in range(3):
            parts=[]
            for key in cells:
                ix=np.flatnonzero(mask&(cell==key)&(y==label))
                assert len(ix)
                w[ix]=1/(len(cells)*len(ix))
                parts.append(counts[train[ix]].mean(0))
            rows.append(np.mean(parts,axis=0))
        templates[p]=np.stack(rows)
    m=np.zeros((len(people),36,3))
    t=np.zeros((len(people),2,36,3))
    for person in [*source,held]:
        rows=np.flatnonzero(people==person)
        donors=[templates[p] for p in source if p!=person]
        assert len(donors)==(9 if person==held else 8)
        m[rows],t[rows]=likelihood(counts[rows],np.mean(donors,axis=0))
    assert np.isfinite(t).all() and np.max(np.abs(t[:,0]-t[:,1]))>1e-6
    w/=w.mean()
    bank={'index':test}
    with np.load(perm) as z:
        orders={o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in base.ORDERS}
    for arm in ARMS:
        x=design(power,m,t,arm)
        scaler=StandardScaler().fit(x[train],sample_weight=w)
        train_x=scaler.transform(x[train])
        model=LogisticRegression(C=1,max_iter=1200,tol=1e-6).fit(train_x,y,sample_weight=w)
        assert model.n_iter_.max()<model.max_iter
        target=scaler.transform(x[test])
        pred=model.predict(target)
        assert np.array_equal(pred[:7],[model.predict(row[None])[0] for row in target[:7]])
        for order,perm_rows in orders.items():
            ix=np.searchsorted(test,perm_rows[people[perm_rows]==held])
            assert np.array_equal(np.sort(ix),np.arange(len(test)))
            restored=np.empty_like(pred);restored[ix]=model.predict(target[ix])
            assert np.array_equal(restored,pred)
            bank[f'{arm}|{order}']=restored
    if smoke:return
    archive=Path(held+'.npz')
    np.savez_compressed(archive,**bank)
    proof=dict(held=held,train_people=source,source_sha256=base.source_manifest(Path('source.sha256')),
               order_sha256=base.ORDER_SHA,outputs_sha256=base.sha(archive),feature_sha256=base.sha(file),
               train_rows_sha256=hashlib.sha256(train.tobytes()).hexdigest(),seconds=time.monotonic()-start,
               guard=dict(prefix=True,reset=True,step_equivalence=True,active=True,finite=True))
    Path(held+'.json').write_text(json.dumps(proof))


if __name__=='__main__':main(sys.argv[1],len(sys.argv)>2)
