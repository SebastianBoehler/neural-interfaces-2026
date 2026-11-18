"""Identify source-only within-trial shared directions and test suppressing their variation."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist,squareform,cdist
from sklearn.utils.extmath import randomized_svd
PRIOR=Path('/mnt/lustre/work/gehler/gfh527/ni26_priv401_source_v1_20261118')
sys.path.insert(0,str(PRIOR))
from source import OLD,HELD,ARMS,features,design_weights,predict,donors,read_training_labels,setup,check,digest


def centered(x,group):
    out=x.copy()
    for g in np.unique(group):
        m=group==g
        out[m]-=out[m].mean(0)
    return out


def main(smoke):
    assets,_,_=setup()
    check(digest(PRIOR/'source.sha256')=='587b2cae97ec18e463c17676ccc79ae4ec7178b60f91977087e68beef9c8de64' and
          digest(OLD/'qualification.json')=='838f4d1d76376a314fb047e6da1d577a7b9b894cd1a3c6afec4d4a6796210dca','lineage')
    qual=json.loads((OLD/'qualification.json').read_text())
    check(digest(OLD/'late.npz')==qual['late_sha256'],'late archive')
    path=assets.CHAIN/'data/proteus_windows.npz'
    with np.load(OLD/'late.npz') as z:
        rows=z['index']
        late=z['X'][:,:,270:450].astype('float64')
    with np.load(path) as z:
        x=z['X']
        person=np.array([p.rsplit('/',1)[-1] for p in z['subject']])
        held=np.flatnonzero(np.isin(person,HELD))
        stream=np.char.add(np.char.add(z['subject'],'|'),z['session'])[rows]
        iface=z['interface'][rows]
        run=np.array([f'{s}|{t}|{r}' for s,t,r in zip(stream,z['task'][rows],z['run'][rows])])
        central=np.char.startswith(z['ch_names'].astype(str),'C')
    check(len(rows)==1894 and len(set(person[rows]))==6 and not np.intersect1d(held,rows).size,'source scope')
    y=read_training_labels(path,rows,len(person))
    if smoke:
        take=run==run[0]
        rows,late,stream,iface,run,y=[a[take] for a in (rows,late,stream,iface,run,y)]
    group=np.char.add(np.char.add(run,'|'),y.astype(str))
    src=features(x[rows])
    query=features(x[rows[:10] if smoke else held])
    avg,std=src.mean(0),np.maximum(src.std(0),1e-6)
    src=(src-avg)/std
    query=(query-avg)/std
    rank=min(96,len(rows)-1)
    _,_,basis=randomized_svd(src,n_components=rank,random_state=401,n_iter=6)
    reduced=src@basis.T
    # Exact whitening after randomized compression.
    ev,eig=np.linalg.eigh(reduced.T@reduced/(len(rows)-1))
    check(ev.min()>1e-8,'degenerate source matrix')
    white=eig/np.sqrt(ev)
    src=reduced@white
    query=query@basis.T@white
    check(np.max(abs(np.cov(src.T)-np.eye(rank)))<1e-6,'whitened metric')
    late-=late.mean(1,keepdims=True)
    late=late[:,central]
    late-=late.mean(-1,keepdims=True)
    fft=abs(np.fft.rfft(late*np.hanning(180),axis=-1))**2
    freq=np.fft.rfftfreq(180,1/120)
    target=np.concatenate([np.log(fft[:,:,(freq>=lo)&(freq<hi)].sum(2).clip(1e-12)) for lo,hi in [(4,8),(8,13),(13,30)]],1)
    target=centered(target,group)
    target/=np.maximum(target.std(0),1e-6)
    residual=centered(src,group)
    directions={}
    spectrum={}
    for a,ix in [('X',np.arange(len(rows))),('D1',donors(group,401)),('D2',donors(group,1401))]:
        u,s,_=np.linalg.svd(residual.T@target[ix]/len(rows),full_matrices=False)
        directions[a]=u[:,:8]
        spectrum[a]=s[:8].tolist()
    directions['M']=directions['X']
    base=squareform(pdist(src,'sqeuclidean'))
    held_dist=cdist(query,src,'sqeuclidean')
    median=np.median(base[np.triu_indices(len(rows),1)])
    outputs={}
    for a in ARMS:
        train,test=base,held_dist
        if a!='O':
            proj=src@directions[a]
            check(abs(np.var(proj,axis=0,ddof=1).sum()-8)<1e-7,'unmatched removed variance')
            sign=1 if a=='M' else -1
            train=base+sign*squareform(pdist(proj,'sqeuclidean'))
            test=held_dist+sign*cdist(query@directions[a],proj,'sqeuclidean')
            check(train.min()>-1e-7 and test.min()>-1e-7,'non-Euclidean metric')
        outputs[a]=predict(np.exp(-train/median),np.exp(-test/median),design_weights(stream,iface,y),np.eye(3)[y])
    check(all(np.max(abs(outputs[a]-outputs['O']))>1e-6 for a in ARMS if a!='O'),'inactive kernel')
    proof=dict(source=digest('source.sha256'),eligible=len(rows),query_labels_opened=False,spectrum=spectrum,
               removed_variance=8,source_dimensions=rank)
    if not smoke:
        np.savez_compressed('members.npz',index=held,**outputs)
        proof['members']=digest('members.npz')
    Path('smoke.json' if smoke else 'source.json').write_text(json.dumps(proof,indent=2))


if __name__=='__main__':
    main(sys.argv[1]=='smoke')
