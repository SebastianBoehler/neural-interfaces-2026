"""Freeze reciprocal pairs of same-class source appearances in native session chunks."""
import hashlib

import numpy as np


def plan(batches,streams,labels,held,seed):
    rng=np.random.default_rng(36036+100*int(held[1:])+seed)
    chosen=[]
    step=0
    for epoch in batches:
        options=[]
        for i,rows in enumerate(epoch):
            for start in range(0,len(rows),32):
                chunk=rows[start:start+32]
                if len(chunk)!=32 or len(set(streams[chunk]))!=1:
                    raise ValueError("native source chunk spans multiple sessions")
                for task in (0,1,2):
                    slots=np.flatnonzero(labels[chunk]==task)+start
                    slots=rng.permutation(slots)
                    _,first=np.unique(rows[slots],return_index=True)
                    slots=slots[np.sort(first)]
                    for a,b in slots[:len(slots)//2*2].reshape(-1,2):
                        options.append((step+i,int(a),int(b)))
        n=sum(map(len,epoch))
        pairs=n//8
        if pairs*8!=n or len(options)<pairs:
            raise ValueError("same-chunk reciprocal pairs cannot cover exact quarter dose")
        for j in sorted(rng.choice(len(options),size=pairs,replace=False).tolist()):
            chosen.append((*options[j],int(rng.choice((-1,1)))))
        step+=len(epoch)
    return np.asarray(chosen,dtype=np.int32)


def audit(batches,streams,tape,labels=None):
    flat=[r for epoch in batches for r in epoch]
    if (tape.ndim!=2 or tape.shape[1]!=4 or tape.dtype!=np.int32 or not len(tape) or
            tape[0,0]<0 or tape[-1,0]>=len(flat) or (np.diff(tape[:,0])<0).any() or
            not np.isin(tape[:,3],(-1,1)).all()):
        raise ValueError("invalid paired training source tape")
    boundaries=np.searchsorted(tape[:,0],np.arange(len(flat)+1))
    seen=np.zeros(len(streams),dtype=np.int64)
    distinct=set()
    for i,rows in enumerate(flat):
        slots=tape[boundaries[i]:boundaries[i+1],1:3]
        if not len(slots):
            continue
        if (slots.min()<0 or slots.max()>=len(rows) or len(np.unique(slots))!=slots.size or
                not np.array_equal(slots[:,0]//32,slots[:,1]//32)):
            raise ValueError("source occurrence duplicate or misplaced within native chunks")
        a,b=rows[slots].T
        if np.any(a==b) or np.any(streams[a]!=streams[b]) or labels is not None and np.any(labels[a]!=labels[b]):
            raise ValueError("paired donor scope crossed source trial, task or session")
        np.add.at(seen,np.concatenate((a,b)),1)
        distinct.update(zip(np.minimum(a,b).tolist(),np.maximum(a,b).tolist()))
    start=0
    for epoch in batches:
        stop=start+len(epoch)
        if 8*(boundaries[stop]-boundaries[start])!=sum(map(len,epoch)):
            raise ValueError("per-epoch finite source exposure changed")
        start=stop
    proof=dict(exposed_rows=sum(map(len,flat)),treated_rows=2*len(tape),distinct_pairs=len(distinct),
               used_source_rows=int(np.count_nonzero(seen)),max_donor_reuse=int(seen.max()),
               tape_rows=hashlib.sha256(tape.astype("<i4",copy=False).tobytes()).hexdigest())
    return boundaries,proof
