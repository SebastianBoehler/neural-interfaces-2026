"""Qualify physical slopes within each instrument under held-person testing."""
import json

import numpy as np
from study import DATA, ROOT, old, panel
from nuisance import effect, expected, matched, model


def fit_predict(low,ratio,duration,people,train,test,w,scale):
    co=model(low,ratio,duration,train,w)
    red=model(low,np.zeros_like(ratio),duration,train,w)
    donor,_=matched(low,people,train,test)
    base=expected(co,low[test])
    predictions=[base+effect(co,ratio[test]),base+effect(co,ratio[donor]),
                 expected(red,low[test])]
    outputs=[(((v-duration[test])/scale)**2).mean(1) for v in predictions]
    return {p:np.array([x[people[test]==p].mean() for x in outputs]) for p in np.unique(people[test])}


def main():
    with np.load(ROOT/'external.npz') as z:
        el,er,ed,ep=z['low'],z['ratio'],z['duration'],z['person']
    with np.load(ROOT/'power.npz') as z:
        pl,pr,pp=z['low'],z['ratio'],np.array([p.split('/')[-1] for p in z['person']])
    with np.load(panel.OLD/'fields.npz') as z:
        pd=z['arr_1']
    with np.load(DATA/'proteus_windows.npz') as z:
        sf=np.char.add(np.char.add(z['subject'],'|'),z['session'])
        iface=z['interface']
    l,r,d,p=np.vstack((pl,el)),np.vstack((pr,er)),np.vstack((pd,ed)),np.r_[pp,ep]
    ext=np.arange(len(pp),len(p))
    pr_scores={}
    panel_fits=[]
    for hold in panel.SOURCES:
        tr=np.flatnonzero(np.isin(pp,list(set(panel.SOURCES)-{hold})))
        te=np.flatnonzero(~np.isin(pp,pp[tr]))
        y=old.source.read_training_labels(DATA/'proteus_windows.npz',tr,len(pp)).astype(int)
        w=old.source.design_weights(sf[tr],iface[tr],y)
        panel_fits.append((tr,w))
        scale=d[tr].std(0)
        within=fit_predict(l,r,d,p,tr,te,w,scale)
        cross=fit_predict(l,r,d,p,ext,te,np.ones(len(ext)),scale)
        for name in within:
            pr_scores.setdefault(name,[]).append(np.r_[within[name],cross[name]])
    ext_scores={}
    for name in sorted(set(ep)):
        tr,te=ext[ep!=name],ext[ep==name]
        scale=d[tr].std(0)
        within=fit_predict(l,r,d,p,tr,te,np.ones(len(tr)),scale)[name]
        cross=np.mean([fit_predict(l,r,d,p,rows,te,w,scale)[name] for rows,w in panel_fits],axis=0)
        ext_scores[name]=np.r_[within,cross]
    out={}
    for key,table in [('proteus',{p:np.mean(v,axis=0) for p,v in pr_scores.items()}),('physionet',ext_scores)]:
        scores=np.array([table[p] for p in sorted(table)])
        rng=np.random.default_rng(413)
        ix=rng.integers(len(table),size=(5000,len(table)))
        report={}
        for field,delta in [('wrong_minus_true',scores[:,1]-scores[:,0]),
                    ('low_only_minus_true',scores[:,2]-scores[:,0]),('cross_minus_within_true',scores[:,3]-scores[:,0]),
                    ('cross_minus_within_increment',(scores[:,5]-scores[:,3])-(scores[:,2]-scores[:,0]))]:
            report[field]=[float(delta.mean()),*np.percentile(delta[ix].mean(1),[10,90]).tolist()]
        out[key]=dict(targets=sorted(table),means=scores.mean(0).tolist(),contrasts=report)
    (ROOT/'within.json').write_text(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
