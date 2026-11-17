"""Full-dimensional structured Gaussian fit; no held/session information in fit."""
import numpy as np
from sklearn.utils.extmath import randomized_svd

D=44239
BLOCKS=tuple([slice(i*861,(i+1)*861) for i in range(51)]+[slice(43911,D)])


def fit(x,y,people):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y);people=np.asarray(people)
    if x.shape!=(len(y),D) or not np.isfinite(x).all() or set(y)!={0,1,2}:
        raise ValueError('invalid full-dimensional source')
    names=sorted(set(people));means=np.stack([[x[(people==p)&(y==c)].mean(0) for c in range(3)] for p in names])
    if not np.isfinite(means).all():raise ValueError('sourceperson missingclass')
    center=means.mean((0,1));residual=x.copy()
    for i,p in enumerate(names):
        for c in range(3):residual[(people==p)&(y==c)]-=means[i,c]
    variance=(residual**2).mean(0)
    for block in BLOCKS:
        typical=float(np.median(variance[block]))
        if typical<=0 or not np.isfinite(typical):raise ValueError('invalid sourceblockvariance')
        variance[block]=.5*variance[block]+.5*typical
    scale=np.sqrt(variance);residual/=scale
    _,singular,v=randomized_svd(residual,n_components=16,n_iter=3,random_state=334)
    eigen=singular**2/len(x);adjust=1/np.sqrt(1+eigen)-1
    model=dict(center=center,scale=scale,v=v,adjust=adjust,people=names)
    z=transform(model,x);m=transform(model,means.reshape(-1,D)).reshape(len(names),3,D)
    global_mean=m.mean(0);q,_=np.linalg.qr((global_mean[:2]-global_mean[2]).T)
    if np.linalg.matrix_rank((global_mean[:2]-global_mean[2]).T)!=2:raise ValueError('semantic span rank!=2')
    offset=m.mean(1);geometry=m-offset[:,None]
    geometry=np.concatenate([global_mean[None],geometry],0)
    geometry-=(geometry@q)@q.T
    model.update(q=q,anchor=global_mean@q,geometry=geometry,offset=np.r_[np.zeros((1,D)),offset],
                 global_geometry=global_mean-(global_mean@q)@q.T)
    if not np.isfinite(z).all() or model['geometry'].shape[2]!=D:raise ValueError('nonfinite/lost dimensions')
    return model


def transform(model,x):
    z=(np.asarray(x)-model['center'])/model['scale']
    return z+((z@model['v'].T)*model['adjust'])@model['v']


def inverse(model,z):
    adjustment=1/(1+model['adjust'])-1
    value=z+((z@model['v'].T)*adjustment)@model['v']
    return value*model['scale']+model['center']
