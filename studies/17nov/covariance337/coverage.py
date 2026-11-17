"""Exact native session chunk permutations, regrouped only for fixed pretext exposure."""
import sys
import numpy as np
from dependencies import BASE,sha

PIN='af0dced9bdcd78781380ff71017d96910f214c545d53c83e121f12d42785fc80'

def plan(d,rows,steps=16):
    path=BASE/'experiments/session_bn.py'
    if sha(path)!=PIN:raise ValueError('native sampler changed')
    sys.path.insert(0,str(BASE/'experiments'))
    from session_bn import SessionNet
    st=d['stream'][rows];streams=list(dict.fromkeys(st))
    order=np.concatenate([rows[st==s] for s in streams])
    obj=object.__new__(SessionNet);obj.group=32;obj.batch=64
    obj._row_sess=np.concatenate([np.full(int((st==s).sum()),j) for j,s in enumerate(streams)])
    rng=np.random.default_rng(0);chunks=[];result=[]
    while len(result)<steps:
        for batch in obj._batches(np.arange(len(order)),rng):
            chunks.extend(list(batch.reshape(-1,32)))
            while len(chunks)>=2 and len(result)<steps:
                result.append(order[np.r_[chunks.pop(0),chunks.pop(0)]])
    result=np.array(result)
    for batch in result:
        for chunk in batch.reshape(2,32):
            if len(set(d['stream'][chunk]))!=1:raise ValueError('cross-session chunk')
    if result.shape!=(steps,64) or not np.isin(result,rows).all():raise ValueError('exposure trace')
    return result
