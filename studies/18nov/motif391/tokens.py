"""Per-window bipolar ordinal transitions and a matched marginal-preserving null."""
import numpy as np

PAIRS = [('C3','CP3'),('C4','CP4'),('Cz','CPz'),('FC3','C3'),('FC4','C4'),('FCz','Cz'),
         ('F3','Fz'),('F4','Fz'),('F5','FC5'),('F6','FC6'),('P3','Pz'),('P4','Pz')]
LAGS = (1,3,8)


def counts(series, lag):
    a,b,c=series[:,:-2*lag],series[:,lag:-lag],series[:,2*lag:]
    # Lehmer rank of three ordered (value,time) pairs; ties favor earlier sample.
    state=2*((a>b).astype(np.int8)+(a>c).astype(np.int8))+(b>c).astype(np.int8)
    first,last=state[:,:-3*lag],state[:,3*lag:]
    perm=np.random.default_rng([391,lag]).permutation(last.shape[1])
    offset=36*np.arange(len(series))[:,None]
    def histogram(end):
        return np.bincount((offset+6*first+end).ravel(),minlength=36*len(series)).reshape(-1,6,6)
    real,null=histogram(last),histogram(last[:,perm])
    assert np.array_equal(real.sum(1),null.sum(1)) and np.array_equal(real.sum(2),null.sum(2))
    return real,null


def features(x, indices):
    s=x[indices[:,0]].astype(np.float64)-x[indices[:,1]]
    power=np.log(np.sqrt(np.mean(s*s,axis=1))+1e-6)
    pairs=[counts(s,d) for d in LAGS]
    return power,np.stack([np.concatenate(p) for p in zip(*pairs)]).astype(np.int16)
