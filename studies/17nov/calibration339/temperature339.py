"""One positive scalar temperature per two-seed ensemble/view, source-held NLL."""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar


def ensemble(logits):
    z=np.stack(logits);lp=z-logsumexp(z,axis=-1,keepdims=True)
    return logsumexp(lp,axis=0)-np.log(len(logits))


def apply(logits,t):
    z=logits/t;return z-logsumexp(z,axis=-1,keepdims=True)


def weights(y,cells):
    w=np.zeros(len(y))
    for c in np.unique(cells):
        mask=cells==c;classes=np.unique(y[mask])
        for k in classes:
            rows=mask&(y==k);w[rows]=1/(len(np.unique(cells))*len(classes)*rows.sum())
    if not np.isclose(w.sum(),1) or np.any(w<=0):raise ValueError('calibrationcell/classweights')
    return w


def fit(banks,y,w,orders):
    # Scheme-equal objective: four within, two full, two runs are NOT8independentpeople.
    ow={o:1/(3*sum(x.startswith(o[:-1]) for x in orders)) for o in orders}
    def objective(logt):return sum(ow[o]*float(np.dot(w,-apply(banks[o],np.exp(logt))[np.arange(len(y)),y])) for o in orders)
    result=minimize_scalar(objective,method='bounded',bounds=(-4.,4.),options={'xatol':1e-10,'maxiter':200})
    if not result.success or not np.isfinite(result.fun):raise ValueError('temperatureoptimizerfailed')
    # Include exact identity: finite optimizer tolerance may not worsen baseline.
    logt=float(result.x) if result.fun<objective(0.) else 0.;t=float(np.exp(logt))
    if any(not np.array_equal(banks[o].argmax(1),apply(banks[o],t).argmax(1)) for o in orders):raise ValueError('temperaturechangedargmax')
    derivative=(objective(logt+1e-5)-objective(logt-1e-5))/2e-5
    return dict(T=t,raw_NLL=objective(0.),fitted_NLL=objective(logt),delta_NLL=objective(logt)-objective(0.),
                bound_hit=abs(logt)>3.99,objective_derivative=derivative,nfev=int(result.nfev),
                max_probability_change=max(float(abs(np.exp(apply(banks[o],t))-np.exp(banks[o])).max()) for o in orders))
