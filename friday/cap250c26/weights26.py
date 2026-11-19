"""Assign complementary frozen-peer weights while preserving each class mass."""

import numpy as np


def multipliers(q,labels):
    assert q.shape==(len(labels),4,3) and np.isfinite(q).all() and (q>=0).all()
    rows=np.arange(len(labels))
    margins=np.log(np.maximum(q[rows,:,labels],1e-6))
    other=np.where(np.arange(3)[None,None,:]==labels[:,None,None],0.,q).max(axis=2)
    margins-=np.log(np.maximum(other,1e-6))
    peers=np.column_stack([np.median(np.delete(margins,k,axis=1),axis=1) for k in range(4)])
    exponent=np.minimum(peers,peers-margins)
    result=np.exp(.5*np.clip(exponent,-np.log(4),np.log(4)))
    assert (result>=.5).all() and (result<=2.).all()
    return result


def rotate(factors,labels):
    rotated=factors.copy()
    for age in range(0,len(labels),20):
        for c in np.unique(labels):
            row=np.flatnonzero((labels[age:age+20]==c))+age
            rotated[row]=np.roll(factors[row],1,axis=0)
            np.testing.assert_array_equal(np.sort(rotated[row],axis=0),np.sort(factors[row],axis=0))
    return rotated


def preserve(factory,weights,labels):
    weights=np.asarray(weights,dtype=np.float32)
    work=weights.astype(np.float64)
    given=factory*work[:,None]
    for c in np.unique(labels):
        selected=labels==c
        given[selected]*=work[selected].sum()/given[selected].sum(0)
    chosen=given.astype(np.float32)
    assert np.isfinite(chosen).all() and (chosen>0).all()
    before=np.array([work[labels==c].sum() for c in np.unique(labels)])
    after=np.array([chosen[labels==c].astype(np.float64).sum(0) for c in np.unique(labels)])
    gap=float(np.abs(before[:,None]/before.sum()-after/after.sum(0)).max())
    assert gap<=2e-7
    exposure=np.array([len(labels)]*4)
    assert np.array_equal((chosen>0).sum(0),exposure)
    return chosen,gap
