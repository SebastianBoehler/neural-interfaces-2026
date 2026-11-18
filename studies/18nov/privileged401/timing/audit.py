"""Source-only temporal pairing test across raw feedback onset."""
import json
import sys
from pathlib import Path
import numpy as np
SRC=Path('/mnt/lustre/work/gehler/gfh527/ni26_priv401_source_v1_20261118')
sys.path.insert(0,str(SRC))
from source import OLD,check,digest,donors,read_training_labels,setup


def vectors(x,idx,run,grp):
    signal=x.astype('float64')
    signal-=signal.mean(axis=1,keepdims=True)
    signal=signal[:,idx]
    signal-=signal.mean(axis=-1,keepdims=True)
    p=abs(np.fft.rfft(signal*np.hanning(180),axis=-1))**2
    f=np.fft.rfftfreq(180,1/120)
    feat=np.concatenate([np.log(p[:,:,(f>=lo)&(f<hi)].sum(2).clip(1e-12)) for lo,hi in [(4,8),(8,13),(13,30)]],axis=1)
    for r in np.unique(run):
        m=run==r
        feat[m]/=np.maximum(feat[m].std(0),1e-6)
    for g in np.unique(grp):
        m=grp==g
        feat[m]-=feat[m].mean(0)
    return feat/np.linalg.norm(feat,axis=1,keepdims=True).clip(1e-12)


def main(smoke):
    assets,_,_=setup()
    qual=json.loads((OLD/'qualification.json').read_text())
    check(digest(OLD/'qualification.json')=='838f4d1d76376a314fb047e6da1d577a7b9b894cd1a3c6afec4d4a6796210dca'
          and 2.50<qual['summary']['feedback_range'][0] and qual['summary']['feedback_range'][1]<3.25,'feedback markers')
    check(digest(OLD/'late.npz')==qual['late_sha256'],'late inputs')
    path=assets.CHAIN/'data/proteus_windows.npz'
    with np.load(OLD/'late.npz') as z:
        rr=z['index']
        late=z['X'][:,:,270:450]
    with np.load(path) as z:
        total=len(z['subject'])
        person=np.array([p.rsplit('/',1)[-1] for p in z['subject'][rr]])
        check(set(person)=={'P04','P05','P06','P07','P08','P10'},'source scope')
        run=np.array([f'{a}|{b}|{c}|{d}' for a,b,c,d in zip(person,z['session'][rr],z['task'][rr],z['run'][rr])])
        wave=z['X'][rr]
        ch=z['ch_names'].astype(str)
    y=read_training_labels(path,rr,total)
    grp=np.char.add(np.char.add(run,'|'),y.astype(str))
    masks={'frontal':[i for i,c in enumerate(ch) if c.startswith('AF') or (c.startswith('F') and not c.startswith('FC'))],
           'central':[i for i,c in enumerate(ch) if c.startswith('C')]}
    check(all(len(idx)>=5 for idx in masks.values()) and not set(masks['frontal'])&set(masks['central']),'ROI identities')
    results={}
    for roi,idx in masks.items():
        vals=[]
        null=[]
        for p in sorted(set(person))[:1 if smoke else 6]:
            m=person==p
            g=grp[m]
            vv=[vectors(x,idx,run[m],g) for x in (wave[m,:,:180],wave[m,:,270:450],late[m])]
            weights=np.zeros(m.sum())
            for group in np.unique(g):
                q=g==group
                weights[q]=1/q.sum()
            weights/=weights.sum()
            sim=lambda order:np.array([np.sum(weights*np.einsum('nf,nf->n',v,vv[2][order])) for v in vv[:2]])
            vals.append(sim(np.arange(m.sum())))
            null.append([sim(donors(g,40100+j)) for j in range(3 if smoke else 256)])
        vals,null=np.stack(vals),np.stack(null)
        delta=vals-null.mean(1)
        rng=np.random.default_rng(401)
        boot=delta[rng.integers(len(vals),size=(4000,len(vals)))].mean(1)
        results[roi]=dict(channels=ch[idx].tolist(),raw=vals.mean(0).tolist(),null90=np.percentile(null.mean(0),90,axis=0).tolist(),
            pairing_advantage=delta.mean(0).tolist(),paired80=np.percentile(boot,[10,90],axis=0).tolist(),
            post_minus_pre=float(np.mean(delta[:,1]-delta[:,0])),post_minus_pre80=np.percentile(boot[:,1]-boot[:,0],[10,90]).tolist(),
            by_person=delta.tolist())
    Path('smoke.json' if smoke else 'result.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main(sys.argv[1]=='smoke')
