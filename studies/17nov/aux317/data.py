"""Source-exclusive §317 splits and immutable input provenance."""
import hashlib
from pathlib import Path

import numpy as np
import torch

ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')
PEOPLE = tuple(f'P{p:02d}' for p in range(1,11))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()


def take(d, people, scale=None):
    rows=np.flatnonzero(np.isin(d['subject'],people))
    x=torch.from_numpy(d['X'][rows].astype(np.float32))
    if scale is None:
        scale=float(x.square().mean().sqrt())
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('invalid source scalar')
    return rows,x/scale,torch.from_numpy(d['y'][rows]).long(),scale


def shin(root, channels):
    path=Path(root)/'data/shin2018/windows.npz'
    with np.load(path,allow_pickle=False) as z:
        if z['X'].shape!=(3120,41,480) or z['classes'].tolist()!=['baseline','WG']:
            raise ValueError('Shin shape/classes differ')
        if not np.array_equal(z['ch_names'],channels) or len(z['recorded'])!=16:
            raise ValueError('Shin channel mask differs')
        if len(np.unique(z['subject']))!=26 or np.bincount(z['y']).tolist()!=[1560,1560]:
            raise ValueError('Shin cohort differs')
        ids=np.array([int(p.split('/')[-1]) for p in z['subject']])
        sr=np.flatnonzero((ids>5)&(z['win']==0)); vr=np.flatnonzero((ids<=5)&(z['win']==0))
        keys=[(str(z['subject'][r]),str(z['block'][r]),int(z['trial'][r])) for r in np.r_[sr,vr]]
        if len(set(keys))!=len(keys):
            raise ValueError('source trial duplicate')
        sx=torch.from_numpy(z['X'][sr].astype(np.float32)); vx=torch.from_numpy(z['X'][vr].astype(np.float32))
        scale=float(sx.square().mean().sqrt())
        if not np.isfinite(scale) or scale<=0:
            raise ValueError('invalid Shin fit scale')
        tensors=(sx/scale,torch.from_numpy(z['y'][sr]).long(),z['subject'][sr],
                 vx/scale,torch.from_numpy(z['y'][vr]).long())
        meta=dict(path=str(path),sha256=sha(path),fit_people=sorted(set(z['subject'][sr])),
                  validation_people=sorted(set(z['subject'][vr])),scale=scale,
                  source_rows_sha256=hashlib.sha256(sr.tobytes()).hexdigest(),
                  recorded=z['recorded'].tolist(),interpolated=z['interpolated'].tolist())
    return tensors,meta


def probabilities(net,x):
    net.eval()
    with torch.no_grad():
        p=torch.cat([net(b).double().softmax(1).cpu() for b in x.split(32)]).numpy()
    if p.shape!=(len(x),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1):
        raise ValueError('invalid probabilities')
    return p
