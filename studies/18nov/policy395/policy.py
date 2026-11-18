"""Fit source-held query policies and lock each outer person's transferred probabilities."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from substrate import DATA,PEOPLE,ORDERS,digest,metadata,orders,outer,read_training_labels,sha,source_bank
from calibration import Policy,fit,forward,sufficient,training

ARMS=('static','native','fixed','learned','mean','donor')


def run(held,smoke):
    torch.set_num_threads(1)
    meta=metadata(); perms=orders()
    target,base=outer(held,meta)
    fitrows=np.flatnonzero(meta['person']!=held)
    labels=np.full(len(meta['person']),-1,int)
    labels[fitrows]=read_training_labels(DATA,fitrows,len(labels))
    assert (labels[target]==-1).all()
    people=sorted(set(meta['person'][fitrows]))
    output={}; proofs={}; prefix=0.; one=0.
    for seed in (0,1):
        bank,owned=source_bank(held,seed,meta,perms)
        arr,owner=training(meta,bank,labels,people)
        learned,rec,trace=fit(arr,owner,seed,False,smoke)
        mean,flat,twin=fit(arr,owner,seed,True,smoke)
        assert np.array_equal(trace,twin)
        fixed=Policy(False)
        proofs[seed]=dict(bodies=owned,trace=digest(trace),learned=rec,mean=flat)
        for o,index in perms.items():
            prob=base[seed]['native|'+o]
            outs={a:np.empty_like(prob) for a in ARMS}
            outs['native']=prob.copy(); outs['static']=base[seed]['static|'+o]
            donor=next(v[4] for v in bank if v[1]==o)
            for stream in np.unique(meta['stream'][target]):
                rows=index[meta['stream'][index]==stream]
                pos=np.searchsorted(target,rows)
                current=prob[pos]
                assert (current>0).all()
                state=sufficient(current)
                alt=sufficient(donor[np.arange(len(rows))%len(donor)])
                for a,model,feat in [('fixed',fixed,state),('learned',learned,state),('mean',mean,state),('donor',learned,alt)]:
                    v=forward(current,feat,model)
                    assert np.allclose(v[0],current[0],atol=1e-12)
                    small=forward(current[:8],sufficient((donor if a=='donor' else current)[:8]),model)
                    prefix=max(prefix,float(abs(small-v[:8]).max()))
                    assert prefix<1e-10
                    outs[a][pos]=v
                assert np.array_equal(state[:1],np.zeros((1,7)))
                # Independent single-window Jacobian accumulation for the first eight.
                b=learned.basis.numpy()
                total=np.zeros(7)
                for i,p in enumerate(current[:8]):
                    one=max(one,float(abs(state[i]-total).max()))
                    j=b.T@(np.diag(p)-p[:,None]*p[None,:])@b
                    total+=np.r_[p@b,j.reshape(-1),1.]
                assert one<1e-10
            for a,v in outs.items():
                assert np.isfinite(v).all()
                output[f'{seed}|{a}|{o}']=v
            assert not np.allclose(outs['learned'],outs['native']) and not np.allclose(outs['mean'],outs['learned'])
    tag=('smoke_' if smoke else '')+held
    path=Path(tag+'.npz');np.savez_compressed(path,index=target,**output)
    doc=dict(held=held,fit=digest(fitrows),target=digest(target),source=sha('source.sha256'),
             outputs=sha(path),held_labels_opened=False,one_step=one,prefix=prefix,proofs=proofs)
    Path(tag+'.json').write_text(json.dumps(doc,indent=2))
    print(tag,one,prefix,flush=True)


if __name__=='__main__':
    run(sys.argv[1],sys.argv[2:]==['smoke'])
