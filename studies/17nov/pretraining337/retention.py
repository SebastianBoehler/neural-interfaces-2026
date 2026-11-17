"""Common source-only ridge(.1) full-target information instrument; no labels."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from common import GROUPS,digest


def mapping(source,query):
    mean=source.mean(0);sd=source.std(0);active=sd>1e-12
    if not active.any() or not np.isfinite(sd).all():raise ValueError('probe inputs invalid')
    x=(source[:,active]-mean[active])/sd[active];q=(query[:,active]-mean[active])/sd[active];n=len(x)
    factor=cho_factor(x.T@x/n+.1*np.eye(x.shape[1]),lower=True)
    weights=cho_solve(factor,q.T).T@x.T/n
    return weights,dict(constant_coordinates=int((~active).sum()),active_coordinates=int(active.sum()),source_mean_sha256=digest(mean),source_sd_sha256=digest(sd))


def oracle(source,held):
    x=source[:24,:12];q=held[:4,:12];y=source[:24,12:20];mu=x.mean(0);sd=x.std(0)
    if (sd<=1e-12).any():raise ValueError('oracle coordinate constant')
    a=(x-mu)/sd;b=(q-mu)/sd;yc=y-y.mean(0)
    coef=np.linalg.lstsq(np.r_[a,np.sqrt(len(a)*.1)*np.eye(12)],np.r_[yc,np.zeros((12,8))],rcond=None)[0]
    weights,_=mapping(x,q);reverse,_=mapping(x[::-1],q)
    error=float(np.abs(weights@yc-b@coef).max());order=float(np.abs(weights@yc-reverse@yc[::-1]).max())
    if max(error,order)>1e-9:raise ValueError('independent ridge/order oracle')
    return dict(oracle_max_abs=error,source_order_max_abs=order)


def prepare(embedding,teacher,n):
    source,held=embedding[:n].astype(np.float64),embedding[n:].astype(np.float64)
    checks=oracle(source,held);weights,meta=mapping(source,held)
    return weights,dict(numerical=checks,normalization=meta,source_target_mean=teacher[:n].mean(0))


def metrics(weights,teacher,n,mean):
    sumsq=np.zeros(3);baseline=np.zeros(3);count=np.zeros(3,int)
    for start in range(0,44239,1024):
        stop=min(start+1024,44239);train=teacher[:n,start:stop].astype(np.float64)-mean[start:stop]
        target=teacher[n:,start:stop].astype(np.float64)-mean[start:stop];prediction=weights@train
        for k,group in enumerate(GROUPS):
            columns=group[(group>=start)&(group<stop)]-start
            sumsq[k]+=np.square(prediction[:,columns]-target[:,columns]).sum();baseline[k]+=np.square(target[:,columns]).sum();count[k]+=len(target)*len(columns)
    return {name:dict(MSE=float(sumsq[k]/count[k]),source_mean_MSE=float(baseline[k]/count[k]),relative_error=float(sumsq[k]/baseline[k])) for k,name in enumerate(('diagonal','offdiagonal','slow'))}
