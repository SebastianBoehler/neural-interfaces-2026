import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from features import CH, ROOT, data393, load


def fit(pairs,rng,rep):
    ds=np.stack([v for v in pairs.values()]).reshape(len(pairs),30,-1)
    centered=ds-ds.mean(1,keepdims=True)
    flat=centered.reshape(-1,ds.shape[-1])
    cov=flat.T@flat/len(flat)
    cov=.5*cov+.5*np.diag(np.diag(cov))
    assert np.linalg.eigvalsh(cov)[0]>0
    signs=np.ones((1,*ds.shape[:2]))
    if rep:
        signs=np.concatenate([signs,rng.choice([-1.,1.],size=(rep,*ds.shape[:2]))])
    mean=np.einsum('spk,pkd->sd',signs,ds)/(ds.shape[0]*ds.shape[1])
    vec=np.linalg.solve(cov,mean.T).T
    return vec/np.linalg.norm(vec,axis=1,keepdims=True)


def auc(y,val):
    gap=val[y==1][:,None]-val[y==0]
    return ((gap>0)+.5*(gap==0)).mean()


def digest(f):
    with open(f,'rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def main(smoke):
    rng=np.random.default_rng(393)
    vectors={}; proofs={}; natives=[]
    for kind,paths in [('A',sorted((ROOT/'data/shin2017b').glob('windows_*.npz'))),
                       ('W',[ROOT/'data/shin2018/windows.npz'])]:
        pairs,rows,channels=data393(kind,paths)
        natives.append(channels)
        assert len(pairs)==(29 if kind=='A' else 26)
        vectors[kind]=fit(pairs,rng,128)
        source={}
        for held in pairs:
            donors={p:v for p,v in pairs.items() if p!=held}
            w=fit(donors,rng,0)[0]
            source[held]=float(np.mean([auc(y,f@w) for p,y,f in rows if p==held]))
        proofs[kind]=source
    assert set(CH)==set.intersection(*natives)
    if smoke:return
    w,a=vectors['W'],vectors['A']
    matrix=np.vstack([w[0]-a[0],w[0]+a[0],w[0],-a[0],w[1:]-a[1:]])
    assert np.isfinite(matrix).all() and matrix.shape==(132,36)
    np.save('directions.npy',matrix)
    for corpus,path in [('Scherer',ROOT/'data/scherer2015/windows.npz'),('PROTEUS',ROOT/'data/proteus_windows.npz')]:
        d,native=load(path,False)
        assert set(CH)==set.intersection(*natives,native)
        v=d['f']@matrix.T
        for i in [0,1,len(v)-1]:assert np.allclose(v[i],matrix@d['f'][i],rtol=1e-12,atol=1e-12)
        np.save(corpus+'.npy',v)
    files=['directions.npy','Scherer.npy','PROTEUS.npy','inputs.sha256','source.sha256']
    Path('before.json').write_text(json.dumps(dict(source_aucs=proofs,hashes={f:digest(f) for f in files})))
if __name__=='__main__':main(sys.argv[1]=='smoke')
