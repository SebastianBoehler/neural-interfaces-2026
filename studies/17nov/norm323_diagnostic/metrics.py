"""Exact original row weights and descriptive probability/argmax decompositions."""
import numpy as np


def weights(y,cell,person,people):
    w=np.zeros(len(y));owners=np.zeros(len(y),int)
    for i,p in enumerate(people):
        cells=np.unique(cell[person==p])
        for c in cells:
            m=cell==c;classes=np.unique(y[m])
            if not np.all(person[m]==p):raise ValueError('crosspersoncell')
            for k in classes:
                rows=m&(y==k);w[rows]=1/(len(cells)*len(classes)*rows.sum());owners[rows]=i
    if (w<=0).any():raise ValueError('unweighted rows')
    return w,owners


def reduce(value,w,owners):
    return np.bincount(owners,weights=np.asarray(value)*w,minlength=10)


def correct(scores,y):return np.asarray(scores).argmax(1)==y


def probability(q,y,w,owners):
    target=q[np.arange(len(y)),y];other=q.copy();other[np.arange(len(y)),y]=-np.inf
    loss=-np.log(np.clip(target,1e-12,1));brier=((q-np.eye(3)[y])**2).sum(1)
    confidence=q.max(1);hit=correct(q,y);ece=np.zeros(10)
    bins=np.minimum((confidence*10).astype(int),9)
    for b in range(10):
        mask=bins==b
        ece+=np.abs(reduce((confidence-hit)*mask,w,owners))
    return {'margin':reduce(target-other.max(1),w,owners),'NLL':reduce(loss,w,owners),
            'Brier':reduce(brier,w,owners),'ECE10':ece}


def decomposition(a,b,y,w,owners):
    old,new=correct(a,y),correct(b,y)
    correction=100*reduce(~old&new,w,owners);destruction=100*reduce(old&~new,w,owners)
    delta=100*reduce(new.astype(float)-old,w,owners)
    if not np.allclose(correction-destruction,delta,atol=1e-11,rtol=0):raise ValueError('correctionidentity')
    return correction,destruction,delta
