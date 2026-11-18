"""Test physical transfer in both directions on different native participants."""
import json

import numpy as np
from study import BASE, DATA, ROOT, old, panel
from nuisance import effect, expected, matched, model


def compare(low,ratio,dur,people,train,test,w):
    coef=model(low,ratio,dur,train,w)
    # Zero ratio column gives the least-squares reduced model on intercept/power alone.
    reduced=model(low,np.zeros_like(ratio),dur,train,w)
    donor,_=matched(low,people,train,test)
    pred=expected(coef,low[test])
    y=dur[test]
    scale=dur[train].std(0)
    assert (scale>0).all()
    mse=lambda x: np.square((x-y)/scale).mean(1)
    true=mse(pred+effect(coef,ratio[test]))
    wrong=mse(pred+effect(coef,ratio[donor]))
    low_only=mse(expected(reduced,low[test]))
    return {p:[float(v[people[test]==p].mean()) for v in (true,wrong,low_only)] for p in np.unique(people[test])}


def main():
    with np.load(ROOT/'power.npz') as z:
        pl,pr,pp=z['low'],z['ratio'],np.array([p.split('/')[-1] for p in z['person']])
    with np.load(ROOT/'external.npz') as z:
        el,er,ep,ed=z['low'],z['ratio'],z['person'],z['duration']
    with np.load(panel.OLD/'fields.npz') as z:
        pd=z['arr_1']
    with np.load(DATA/'proteus_windows.npz') as z:
        stream=np.char.add(np.char.add(z['subject'],'|'),z['session'])
        iface=z['interface']
    low,ratio,duration,people=np.vstack((pl,el)),np.vstack((pr,er)),np.vstack((pd,ed)),np.r_[pp,ep]
    ex=np.arange(len(pp),len(people))
    assert len(set(ep))==20 and len(set(pp))==10 and not set(pp)&set(ep)
    panels=[]
    for held in panel.SOURCES:
        tr=np.flatnonzero(np.isin(pp,list(set(panel.SOURCES)-{held})))
        y=old.source.read_training_labels(DATA/'proteus_windows.npz',tr,len(pp)).astype(int)
        w=old.source.design_weights(stream[tr],iface[tr],y)
        panels.append(compare(low,ratio,duration,people,tr,ex,w))
    forward={p:np.mean([s[p] for s in panels],axis=0) for p in set(ep)}
    rev=compare(low,ratio,duration,people,ex,np.arange(len(pp)),np.ones(len(ex)))
    out={}
    for name,table in [('proteus_to_physionet',forward),('physionet_to_proteus',rev)]:
        data=np.array([table[p] for p in sorted(table)])
        delta=data[:,1:]-data[:,[0]]
        ix=np.random.default_rng(413).integers(len(data),size=(5000,len(data)))
        out[name]=dict(targets=sorted(table),loss_true_wrong_low=data.tolist(),
            mean_loss=data.mean(0).tolist(),mean_reduction=delta.mean(0).tolist(),
            interval80=np.percentile(delta[ix].mean(1),[10,90],axis=0).T.tolist())
    (ROOT/'transfer.json').write_text(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
