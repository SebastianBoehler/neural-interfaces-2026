"""Fit conditional dynamics and an independent readout without the outer held person's labels."""
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import ARMS,DATA,OLD,PEOPLE,SEEDS,digest,metadata,orders,sha
assert sha(OLD/'source_labels.py')=='35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc'
sys.path.insert(0,str(OLD))
from source_labels import read_training_labels


def subsets(rows,groups,y,seed):
    rng=np.random.default_rng(seed)
    template=[]
    read=[]
    for g in np.unique(groups):
        for c in range(3):
            selected=rows[(groups==g)&(y==c)]
            shuffled=rng.permutation(selected)
            k=2*len(selected)//3
            assert k>0 and k<len(selected)
            template.extend(shuffled[:k])
            read.extend(shuffled[k:])
    return np.sort(template),np.sort(read)


def weights(rows,cells,y):
    w=np.zeros(len(rows))
    for g in np.unique(cells[rows]):
        for c in range(3):
            mask=(cells[rows]==g)&(y[rows]==c)
            assert mask.any()
            w[mask]=1/mask.sum()
    return w/w.mean()


def profile(gram,perm_xy,train,y,w,arm):
    # Physical units cancel in the ridge parameter and the source-only readout scaling.
    xx=gram[:,:48,:48]
    xy=perm_xy if arm=='P' else gram[:,:48,48:]
    yy=gram[:,48:,48:].diagonal(axis1=1,axis2=2)
    features=[]
    for c in (range(1) if arm=='C' else range(3)):
        mask=np.ones(len(train),bool) if arm=='C' else y[train]==c
        rows=train[mask]
        a=np.average(xx[rows],axis=0,weights=w[mask]).astype(np.float64)
        b=np.average(xy[rows],axis=0,weights=w[mask])
        coef=np.linalg.solve(a+.02*a.trace()/48*np.eye(48),b).astype(np.float32)
        energy=yy-2*(xy*coef).sum(1)+((xx@coef)*coef).sum(1)
        assert (energy>0).all() and np.isfinite(energy).all()
        features.append(np.log(energy.astype(np.float64)))
    f=np.stack(features,1)
    mean=f.mean(1)
    return np.concatenate((mean,(f-mean[:,None]).reshape(len(f),-1)),1)


def centered(logits,other,pop):
    count=np.arange(len(logits))[:,None]
    hist=other[np.arange(len(logits))%len(other)]-pop
    prior=np.concatenate((np.zeros_like(hist[:1]),hist.cumsum(0)[:-1]))
    return logits-prior/(20+count)


def run(held,form,smoke):
    meta=metadata()
    receipts=json.loads(Path('inputs.json').read_text())
    assert receipts['source']==sha('source.sha256')
    for k in ('gram','perm_xy'):
        assert sha(k+'.npy')==receipts[k]
    gram=np.load('gram.npy',mmap_mode='r')
    perm=np.load('perm_xy.npy',mmap_mode='r')
    fit=np.flatnonzero(meta['person']!=held)
    test=np.flatnonzero(meta['person']==held)
    y=np.full(2655,-1,int)
    y[fit]=read_training_labels(DATA,fit,len(y))
    assert (y[test]==-1).all() and set(meta['person'][fit])==set(PEOPLE)-{held}
    cells=np.char.add(np.char.add(meta['stream'],'|'),meta['interface'])
    donors=sorted(set(meta['stream'][fit]))
    donor_mask=meta['stream']==donors[0]
    output={}
    seeds={}
    for seed in SEEDS[:1] if smoke else SEEDS:
        train,read=subsets(fit,cells[fit],y[fit],seed)
        assert not np.intersect1d(train,read).size and np.array_equal(np.sort(np.r_[train,read]),fit)
        w=weights(train,cells,y)
        rw=weights(read,cells,y)
        sources={}
        for arm in ('D','C','P'):
            fs=[]
            for b in range(3):
                halves=[0,1] if form=='half' else [None]
                for h in halves:
                    g=gram[:,:,b].mean(1) if h is None else gram[:,h,b]
                    q=perm[:,:,b].mean(1) if h is None else perm[:,h,b]
                    fs.append(profile(g,q,train,y,w,arm))
            f=np.concatenate(fs,axis=1)
            center=np.average(f[read],axis=0,weights=rw)
            sd=np.sqrt(np.average((f[read]-center)**2,axis=0,weights=rw))
            f=(f-center)/np.where(sd>1e-8,sd,1.)
            model=LogisticRegression(C=.05,max_iter=500,tol=.0001).fit(f[read],y[read],sample_weight=rw)
            assert (model.n_iter_<500).all()
            sources[arm]=model.decision_function(f)
        assert not np.allclose(sources['D'],sources['P']) and not np.allclose(sources['D'],sources['C'])
        pop=np.average(sources['D'][read],axis=0,weights=rw)
        for o,index in orders().items():
            rows=index[np.isin(index,test)]
            donor=sources['D'][index[donor_mask[index]]]
            for arm in ARMS:
                v=np.empty((len(test),3))
                for s in np.unique(meta['stream'][test]):
                    rr=rows[meta['stream'][rows]==s]
                    base=sources.get(arm,sources['D'])[rr]
                    if arm in ('A','donor'):
                        other=base if arm=='A' else donor
                        val=centered(base,other,pop)
                        prefix=centered(base[:7],other[:7],pop)
                        assert np.allclose(val[:7],prefix) and np.array_equal(val[0],base[0])
                    else:
                        val=base
                    v[np.searchsorted(test,rr)]=val
                assert np.isfinite(v).all()
                output[f'{seed}|{arm}|{o}']=v.argmax(1).astype(np.int8)
        seeds[seed]=dict(template=digest(train),readout=digest(read),source_count=len(train),readout_count=len(read))
    tag=('smoke_' if smoke else '')+form+'_'+held
    np.savez_compressed(tag+'.npz',index=test,**output)
    Path(tag+'.json').write_text(json.dumps(dict(held=held,source=sha('source.sha256'),inputs=sha('inputs.json'),
        fit=digest(fit),test=digest(test),seeds=seeds,outputs=sha(tag+'.npz'),held_labels_opened=False),indent=2))


if __name__=='__main__':
    run(sys.argv[1],sys.argv[2],sys.argv[3:]==['smoke'])
