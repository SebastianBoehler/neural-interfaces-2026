"""Pair original and corrected panels, retaining person and order dependence."""
import json
import sys
from pathlib import Path

import numpy as np
from study import ARMS, DATA, PANELS, ROOT, digest, old, panel, require

# Import immutable §410 I/O by path; the local scorer has the same module name.
import importlib.util
spec=importlib.util.spec_from_file_location('source_score410',PANELS/'score.py')
ancestry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ancestry)


def read(people):
    with np.load(old.source.INPUT/'inputs.npz') as z:
        orders=z['orders'].tolist()
    inputs,proof=[],[]
    for i,p in enumerate(panel.SOURCES):
        arms={}
        held=panel.DEV+(p,)
        for folder,names,prefix in ((PANELS/str(i),old.ARMS,'old'),(ROOT/str(i),ARMS,'new')):
            ancestry.ARMS=names
            ix,outputs,seals=ancestry.read(folder,folder,held,people,orders)
            proof.append(seals)
            for fam, preds in outputs.items():
                for name,val in preds.items():
                    arms[fam,prefix+name]=val
        np.testing.assert_array_equal(arms['full','oldZ'],arms['full','newZ'])
        inputs.append((held,ix,arms))
    return inputs,proof


def measure(inputs,y,iface,session,people,first):
    out={}
    for held,ix,arms in inputs:
        for key,pred in arms.items():
            per=[]
            for p in held:
                s=(people[ix]==p)
                if first:
                    s&=session[ix]=='1'
                cells=[(iface[ix]==f)&(session[ix]==ss)&(y[ix]==c)&s
                       for f in set(iface[ix][s]) for ss in set(session[ix][s]) for c in range(3)]
                require(all(np.any(rr) for rr in cells),'empty cell')
                per.append(np.mean([(pred[:,rr]==y[ix][rr]).mean(1)*100 for rr in cells],0))
            out.setdefault(key,[]).append(per)
    return {k:np.array(v) for k,v in out.items()}


def summary(a,weights):
    order_mean=ancestry.mean_order
    s,d=a[:,4],a[:,:4]
    source=order_mean(s)
    dev=order_mean(d).mean(0)
    person=np.r_[source,dev]
    rng=np.random.default_rng(413)
    n=5000
    si=rng.integers(6,size=(n,6))
    di=rng.integers(4,size=(n,4))
    bs=np.zeros((n,6))
    bd=np.zeros((n,4))
    for g in ancestry.GROUPS:
        os=rng.choice(g,size=(n,6,len(g)))
        od=rng.choice(g,size=(n,4,len(g)))
        bs+=s[si[:,:,None],os].mean(2)/3
        bd+=d[si[:,:,None,None],di[:,None,:,None],od[:,None,:,:]].mean((1,3))/3
    w=np.asarray(weights)
    weighted=((bs*w[si]).sum(1)+(bd*w[6+di]).sum(1))/(w[si].sum(1)+w[6+di].sum(1))
    def pack(point,draws):
        return [float(point),*np.percentile(draws,[10,90]).tolist()]
    return dict(source=pack(source.mean(),bs.mean(1)),development=pack(dev.mean(),bd.mean(1)),
                gap=pack(dev.mean()-source.mean(),(bd.mean(1)-bs.mean(1))),
                official=pack(np.average(person,weights=w),weighted),
                people=dict(zip(panel.SOURCES+panel.DEV,person.tolist())))


def main():
    require((ROOT/'replay.log').read_text().splitlines()[-1]=='OK','replay')
    require(digest(PANELS/'score.py')=='9a21e804e2bc82674978a81c2368527451feb9c30730f70532eafd16bb716c9e','old scorer')
    with np.load(DATA/'proteus_windows.npz') as z:
        people=np.array([p.split('/')[-1] for p in z['subject']])
        iface,session=z['interface'],z['session']
    inputs,proof=read(people)
    (ROOT/'proof.json').write_text(json.dumps(proof))
    with np.load(DATA/'proteus_windows.npz') as z:
        y=z['y']
    report={}
    for first in (False,True):
        measured=measure(inputs,y,iface,session,people,first)
        weights=[1 if first else len(set(session[people==p])) for p in panel.SOURCES+panel.DEV]
        for fam in ('whole','full'):
            def m(arm):
                return measured[fam,arm]
            for name,val in [('intact',m('oldX')),('shuffled',m('oldD')),('true',m('newX')),('wrong',m('newW')),
                    ('original_gap',m('oldX')-m('oldD')),('true_gap',m('newX')-m('oldD')),
                    ('wrong_gap',m('newW')-m('oldD')),('shrink',m('oldX')-m('newX')),
                    ('extra_shrink',m('newW')-m('newX')),('true_minus_wrong',m('newX')-m('newW'))]:
                report[f'{fam}/{int(first)}/{name}']=summary(val,weights)
            if fam=='full':
                report[f'full/{int(first)}/frozen']=summary(m('oldZ'),weights)
                report[f'full/{int(first)}/true_minus_frozen']=summary(m('newX')-m('oldZ'),weights)
    np.testing.assert_allclose(report['full/0/frozen']['official'][0],76.2793,atol=.0001,rtol=0)
    np.testing.assert_allclose(report['whole/0/original_gap']['gap'][0],3.372,atol=.001,rtol=0)
    external=[json.loads((ROOT/str(i)/'external.json').read_text()) for i in range(6)]
    persons=sorted(external[0])
    losses=np.mean([[b[p] for p in persons] for b in external],axis=0)
    external_summary={}
    for j,name in [(0,'center_minus_true'),(2,'wrong_minus_true')]:
        delta=losses[:,j]-losses[:,1]
        draws=np.random.default_rng(413).choice(delta,(5000,len(delta))).mean(1)
        external_summary[name]=[float(delta.mean()),*np.percentile(draws,[10,90]).tolist()]
    match=[json.loads((ROOT/str(i)/'source.json').read_text())['donor_median'] for i in range(6)]
    result=dict(summary=report,external=external_summary,external_people=persons,donor_median=match,
                seed=413,source=digest(ROOT/'source.sha256'),score=digest(__file__),proof=digest(ROOT/'proof.json'))
    (ROOT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
