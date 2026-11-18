"""Three-person out-of-body source fit. No incumbent scores used by the calibrator."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax

from inputs384 import CHAIN, COHORT, ORDERS, check


def read(bank,order,include_j):
    j,k = bank[('J_real' if include_j else 'K',order)],bank['K',order]
    avg = (j+k)*.5
    log = np.log(np.maximum(avg,1e-12))
    inputs = []
    for p in (j,k):
        l = np.log(np.maximum(p,1e-12))
        inputs.append(l-l.mean(1,keepdims=True))
    return avg,log,np.column_stack(inputs)


def fit(held,rows,meta,bank,reader,include_j):
    per_order = np.array([1/12]*4+[1/6]*4)
    x,l,y,weights = [],[],[],[]
    sources = sorted(set(COHORT)-{held})
    for p in sources:
        r = rows[p]
        check(not np.intersect1d(r,rows[held]).size,'training collision')
        target = reader(CHAIN/'data/proteus_windows.npz',r,len(meta['person']))
        group = np.char.add(meta['interface'][r],target.astype(str))
        _,g,n = np.unique(group,return_inverse=True,return_counts=True)
        check(len(n)==6, 'source cell/class count')
        for i,o in enumerate(ORDERS):
            _,log,feat = read(bank[p],o,include_j)
            x.append(feat);l.append(log);y.append(target)
            weights.append(per_order[i]/(len(sources)*len(n)*n[g]))
    x,l,y,w = [np.concatenate(t) for t in (x,l,y,weights)]
    check(np.isclose(w.sum(),1),'source group mass')
    m = w@x
    sd = np.sqrt(w@((x-m)**2)).clip(1e-4)
    x = np.column_stack(((x-m)/sd,np.ones(len(y))))
    eye = np.eye(3)[y]
    def objective(params):
        b = params.reshape(x.shape[1],3)
        q = softmax(l+x@b,axis=1)
        return -w@np.log(q[np.arange(len(y)),y]) + .05*np.sum(b*b), ((x.T*w)@(q-eye)+.1*b).ravel()
    sol = minimize(objective,np.zeros(x.shape[1]*3),method='L-BFGS-B',jac=True,
                   options={'gtol':1e-9,'ftol':1e-13,'maxiter':150})
    def temp(logt):
        q = softmax(l/np.exp(logt),axis=1)
        return -w@np.log(np.maximum(q[np.arange(len(y)),y],1e-300))
    t = minimize_scalar(temp,method='bounded',bounds=(-3,3))
    check(sol.success and t.success and np.isfinite(sol.x).all(), 'calibration optimizer failed')
    coeff = sol.x.reshape(x.shape[1],3)
    return dict(source=sources,mean=m.tolist(),sd=sd.tolist(),coef=coeff.tolist(),T=float(np.exp(t.x)))


def evaluate(p,bank,order,include_j):
    avg,log,x = read(bank,order,include_j)
    x = np.column_stack(((x-p['mean'])/p['sd'],np.ones(len(x))))
    return avg, softmax(log/p['T'],axis=1),softmax(log+x@np.asarray(p['coef']),axis=1)
