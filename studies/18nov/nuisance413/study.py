"""Run source-only §413 fits against immutable five-person panels."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from nuisance import effect, expected, matched, model, spectrum

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/lustre/work/gehler/gfh527')
PANELS = BASE/'ni26_panel410_v1_20261118'
sys.path.insert(0,str(PANELS))
import panel

old = panel.old
require, digest = panel.check, panel.digest
require(digest(PANELS/'panel.py')=='980b7569ea30af47e98d3acacc0bbb4a3e107fba20a0f7289d52ebbce240706d','actor lineage')
DATA = BASE/'neural-interfaces-2026/data'
ARMS = ('X','W')


def pairs(names):
    return np.array([(names.index(a),names.index(b)) for a,b in old.parent.PAIRS])


def extract(external,smoke):
    files = [DATA/'proteus_windows.npz'] if not external else [
        DATA/'physionetmi'/f'windows_{a}-{a+9}.npz' for a in (1,11)]
    output, labels, duration = [],[],[]
    for filename in files:
        with np.load(filename) as z:
            pair = pairs(z['ch_names'].tolist())
            x, person = z['X'], z['subject']
            rows = np.arange(len(x))
            if external:
                require(len(z['interpolated'])==0,'interpolated external electrodes')
                rows = np.concatenate([np.flatnonzero(person==p)[:40] for p in np.unique(person)])
            for i in rows[:2] if smoke else rows:
                fields = spectrum(x[i],pair)
                output.append(fields)
                labels.append(person[i])
                if external:
                    duration.append(old.features(x[i],pair)[1])
                if smoke:
                    np.testing.assert_allclose(spectrum(-x[i],pair),fields,atol=0,rtol=0)
    data=np.array(output)
    require(np.isfinite(data).all() and np.ptp(data[:,1])>0,'inactive proxy')
    if not smoke:
        np.savez(ROOT/('external.npz' if external else 'power.npz'),low=data[:,0],ratio=data[:,1],
                 person=labels,duration=duration)


def fit(i,smoke):
    from scipy.spatial.distance import pdist, squareform, cdist
    from sklearn.preprocessing import StandardScaler
    with np.load(DATA/'proteus_windows.npz') as z:
        people = np.array([p.split('/')[-1] for p in z['subject']])
        stream = np.char.add(np.char.add(z['subject'],'|'),z['session'])
        interface = z['interface']
    with np.load(ROOT/'power.npz') as z:
        low, ratio = z['low'], z['ratio']
        np.testing.assert_array_equal(people,[p.split('/')[-1] for p in z['person']])
    held = panel.DEV+(panel.SOURCES[i],)
    test = np.flatnonzero(np.isin(people,held))
    train = np.flatnonzero(~np.isin(people,held))
    require(set(people[train])==set(panel.SOURCES)-{panel.SOURCES[i]},'train leak')
    labels = old.source.read_training_labels(DATA/'proteus_windows.npz',train,len(people)).astype(int)
    w = old.source.design_weights(stream[train],interface[train],labels)
    path = panel.OLD/'fields.npz'
    require(digest(path)=='a53cba78199b91298f8420e4952d4786267262238e814f04afc00a2371ac51bb','duration lineage')
    with np.load(path) as z:
        amp, dur = z['arr_0'],z['arr_1']
    coefs = model(low,ratio,dur,train,w)
    center = np.average(ratio[train],axis=0,weights=w)
    idx,dist = matched(low,people,train,np.arange(len(low)))
    results={}
    for name, r in [('X',ratio),('W',ratio[idx])]:
        matrix = np.column_stack((amp,dur-effect(coefs,r-center)))
        scale = StandardScaler().fit(matrix[train],sample_weight=w)
        a,b=scale.transform(matrix[train]),scale.transform(matrix[test])
        if smoke:
            b=b[:5]
        d=pdist(a,'sqeuclidean')
        width=np.median(d)
        results[name]=old.source.predict(np.exp(-squareform(d)/width),np.exp(-cdist(b,a,'sqeuclidean')/width),
                                         w,np.eye(3)[labels])
    require(np.isfinite(list(results.values())).all() and np.max(abs(results['X']-results['W']))>.00001,'inactive')
    if smoke:
        return
    np.savez_compressed('members.npz',index=test,**results)
    Path('source.json').write_text(json.dumps(dict(source=digest('source.sha256'),members=digest('members.npz'),
        held=held,train=sorted(set(people[train])),query_labels_opened=False,donor_median=float(np.median(dist[test])))))
    with np.load(ROOT/'external.npz') as z:
        el,er,ep,ed=z['low'],z['ratio'],z['person'],z['duration']
    n=len(low)
    donor,_=matched(np.vstack((low,el)),np.r_[people,ep],train,np.arange(n,n+len(el)))
    base=expected(coefs,el)+effect(coefs,np.broadcast_to(center,er.shape))
    true=base+effect(coefs,er-center)
    wrong=base+effect(coefs,ratio[donor]-center)
    # Longest duration is in seconds; sustained fraction is unitless. Scale by source spread.
    scale=np.maximum(dur[train].std(0),1e-8)
    loss=[(((pred-ed)/scale)**2).mean(1) for pred in (base,true,wrong)]
    require(len(set(ep))==20 and np.isfinite(loss).all(),'external subjects')
    Path('external.json').write_text(json.dumps({p:[float(v[ep==p].mean()) for v in loss] for p in np.unique(ep)}))


if __name__=='__main__':
    mode=sys.argv[1]
    i=int(sys.argv[2]) if len(sys.argv)>2 else 0
    if mode in ('smoke','extract','external'):
        extract(mode=='external',mode=='smoke')
        if mode=='smoke':
            extract(True,True)
    else:
        os.chdir(ROOT/str(i))
        if mode in ('fit','fit-smoke'):
            fit(i,mode=='fit-smoke')
        else:
            require(mode in ('order','actor'),'mode')
            old.ARMS=ARMS
            panel.actor(mode=='actor',int(sys.argv[3]))
