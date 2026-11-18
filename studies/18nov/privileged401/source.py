"""Train from official EEG with source-only cross-person late-interval soft targets."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist,pdist
from scipy.special import softmax

BASE=Path('/mnt/lustre/work/gehler/gfh527')
OLD=BASE/'ni26_crops347_prepare_v2_20261117'
STAGE=BASE/'ni26_turn386_v1_20261118'
sys.path.insert(0,str(STAGE))
from worker import setup,digest,check,INPUT,SCHEDULE
READER=BASE/'ni26_s306_expert_cpu_v2_20261116'
sys.path.insert(0,str(READER))
from source_labels import read_training_labels
HELD=('P01','P02','P03','P09')
ARMS=('O','M','X','D1','D2')


def features(x):
    n=x.shape[-1]
    f=np.fft.rfftfreq(n,1/120)
    tri=np.triu_indices(41,1)
    out=[]
    for block in np.array_split(x,max(1,len(x)//32)):
        v=block.astype('float64')
        v-=v.mean(axis=1,keepdims=True)
        v-=v.mean(axis=-1,keepdims=True)
        fft=np.fft.rfft(v*np.hanning(n),axis=-1)
        cols=[]
        for lo,hi in [(4,8),(8,12),(12,18),(18,25),(25,36)]:
            spec=fft[:,:,(f>=lo)&(f<hi)]
            cov=(spec@spec.conj().transpose(0,2,1)).real/(n*np.sum(np.hanning(n)**2))
            var=np.diagonal(cov,axis1=1,axis2=2).clip(1e-15)
            corr=cov/np.sqrt(var[:,:,None]*var[:,None,:])
            cols.extend((np.log(var),corr[:,tri[0],tri[1]]))
        out.append(np.concatenate(cols,axis=1))
    return np.concatenate(out)


def kernel(x,q):
    mean=x.mean(0)
    std=np.maximum(x.std(0),1e-5)
    a=(x-mean)/std
    b=(q-mean)/std
    ds=pdist(a,'sqeuclidean')
    scale=np.median(ds)
    check(scale>0,'kernel scale')
    from scipy.spatial.distance import squareform
    return np.exp(-squareform(ds)/scale),np.exp(-cdist(b,a,'sqeuclidean')/scale)


def design_weights(stream,interface,y):
    keys=[(str(s),str(f),int(t)) for s,f,t in zip(stream,interface,y)]
    from collections import Counter
    counts=Counter(keys)
    w=np.array([1/counts[k] for k in keys])
    return w/w.mean()


def predict(k,q,w,targets):
    alpha=solve(k+np.diag(1/w),targets,assume_a='pos')
    if targets.ndim==2:
        return softmax(4*(q@alpha),axis=1)
    raise ValueError('expected target matrix')


def donors(keys,seed):
    rng=np.random.default_rng(seed)
    ix=np.arange(len(keys))
    for group in sorted(set(keys)):
        rr=np.flatnonzero(keys==group)
        check(len(rr)>1,'no admissible donor')
        order=rng.permutation(rr)
        ix[order]=np.roll(order,1)
    check(not np.any(ix==np.arange(len(ix))) and np.array_equal(keys,keys[ix]),'same-run/class donor')
    return ix


def main(smoke):
    assets,_,_=setup()
    check(digest(OLD/'qualification.json')=='838f4d1d76376a314fb047e6da1d577a7b9b894cd1a3c6afec4d4a6796210dca','crop qualification')
    qual=json.loads((OLD/'qualification.json').read_text())
    check(qual['passed'] and not qual['query_transformed'] and not qual['held_labels_opened'] and
          digest(OLD/'late.npz')==qual['late_sha256']=='c1dcb7979a53525ac5edbff81dcdbe01c9fdc2d40ec166b182ed48da97cd6042','privileged archive')
    check(digest(READER/'source_labels.py')=='35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc','source label reader')
    path=assets.CHAIN/'data/proteus_windows.npz'
    with np.load(OLD/'late.npz') as z:
        rows,late=z['index'],z['X']
    with np.load(path) as d:
        x=d['X']
        person=np.array([p.rsplit('/',1)[-1] for p in d['subject']])
        held=np.flatnonzero(np.isin(person,HELD))
        stream=np.char.add(np.char.add(d['subject'],'|'),d['session'])[rows]
        iface=d['interface'][rows]
        run=np.array([f'{s}|{t}|{r}' for s,t,r in zip(stream,d['task'][rows],d['run'][rows])])
    check(len(rows)==1894 and len(set(person[rows]))==6 and not np.intersect1d(rows,held).size and late.shape==(1894,41,480),'rows')
    diff=float(np.max(abs(x[rows,:,-180:]-late[:,:,:180])))
    check(diff<=2e-4,f'overlap raw mismatch {diff}')
    y=read_training_labels(path,rows,len(person))
    y=np.asarray(y,dtype=int)
    keys=np.char.add(np.char.add(run,'|'),y.astype(str))
    ds=[donors(keys,k) for k in (401,1401)]
    if smoke:
        v=features(late[:36,:,180:])
        k,q=kernel(v[:24],v[24:])
        p=predict(k,q,np.ones(24),np.eye(3)[y[:24]])
        check(np.isfinite(p).all() and np.max(np.ptp(p,axis=0))>1e-6,'real source kernel inactive')
        Path('smoke.json').write_text(json.dumps(dict(overlap=diff,source=digest('source.sha256'))))
        return
    check(json.loads(Path('smoke.json').read_text())['source']==digest('source.sha256'),'smoke seal')
    suffix=features(late[:,:,180:])
    teacher=np.zeros((len(rows),3))
    people=person[rows]
    for p in np.unique(people):
        train=people!=p
        k,q=kernel(suffix[train],suffix[~train])
        w=design_weights(stream[train],iface[train],y[train])
        teacher[~train]=predict(k,q,w,np.eye(3)[y[train]])
    mean=np.empty_like(teacher)
    for g in np.unique(keys):
        m=keys==g
        mean[m]=teacher[m].mean(0)
    one=np.eye(3)[y]
    targets={'O':one,'M':.7*one+.3*mean,'X':.7*one+.3*teacher}
    for a,idx in zip(('D1','D2'),ds):
        targets[a]=.7*one+.3*teacher[idx]
    official=features(x[rows])
    heldfeat=features(x[held])
    k,q=kernel(official,heldfeat)
    w=design_weights(stream,iface,y)
    outs={a:predict(k,q,w,t) for a,t in targets.items()}
    check(all(np.max(abs(outs['X']-outs[a]))>1e-6 for a in ARMS if a!='X'),'inactive contrast')
    audit={p:float(np.average(teacher[people==p].argmax(1)==y[people==p],weights=design_weights(stream[people==p],iface[people==p],y[people==p]))) for p in np.unique(people)}
    np.savez_compressed('members.npz',index=held,**outs)
    Path('source.json').write_text(json.dumps(dict(members=digest('members.npz'),source=digest('source.sha256'),overlap=diff,
        teacher_by_person=audit,eligible=len(rows),query_labels_opened=False,rows=hashlib.sha256(rows.tobytes()).hexdigest()),indent=2))


if __name__=='__main__':
    main(sys.argv[1]=='smoke')
