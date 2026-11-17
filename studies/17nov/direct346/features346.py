"""Direct signed slow waveform and high-band power, current window only."""
import sys
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from common346 import BASE,digest,sha

sys.path.insert(0,str(BASE/'experiments'))
import features
if sha(features.__file__)!=sha(BASE/'experiments/features.py'):raise ValueError('wrong feature owner')


def direct(raw):
    if raw.ndim!=3 or raw.shape[1:]!=(41,480) or not np.isfinite(raw).all():raise ValueError('official current-window schema')
    slow=features.slow_waveform(raw,cutoff=1.,step=60).astype(np.float64)
    high=features.bandpass(raw,40.,58.).astype(np.float64)
    result=np.c_[slow,np.log(np.maximum(np.mean(high*high,axis=-1),1e-12))]
    if result.shape!=(len(raw),369) or not np.isfinite(result).all():raise ValueError('direct summary schema')
    return result


def shuffled(fit,d,run,extra):
    order=np.arange(len(fit));rng=np.random.default_rng(346)
    keys=np.array([f'{d["stream"][r]}|{run[r]}' for r in fit])
    for key in np.unique(keys):
        group=np.flatnonzero(keys==key)
        if len(group)<2:raise ValueError('cannot derange source run')
        order[group]=np.roll(group,int(rng.integers(1,len(group))))
    if np.any(order==np.arange(len(fit))) or not np.array_equal(np.sort(order),np.arange(len(fit))) or not np.array_equal(keys,keys[order]):raise ValueError('source/run derangement failed')
    if np.array_equal(extra,extra[order]):raise ValueError('derangement inactive')
    return extra[order],order


def fit_head(x,y):
    scale=StandardScaler().fit(x);z=scale.transform(x)
    if np.any(scale.var_<=1e-20) or not np.isfinite(z).all():raise ValueError('constant/nonfinite source coordinate')
    model=LogisticRegression(C=.01,max_iter=2000,tol=1e-5,solver='lbfgs')
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);model.fit(z,y)
    if set(model.classes_)!={0,1,2} or max(model.n_iter_)>=2000:raise ValueError('readout convergence/classes failed')
    return scale,model,dict(scaler_mean_sha256=digest(scale.mean_),scaler_scale_sha256=digest(scale.scale_),
        coefficient_sha256=digest(model.coef_),iterations=model.n_iter_.tolist(),width=x.shape[1])


def predict(head,x):
    scale,model,_=head;p=model.predict_proba(scale.transform(x))
    if p.shape!=(len(x),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-12):raise ValueError('invalid readout posterior')
    return p
