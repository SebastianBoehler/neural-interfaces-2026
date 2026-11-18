"""Row-local bipolar spectrum and source-estimated nuisance correction."""
import numpy as np
from scipy.spatial import cKDTree


def spectrum(x, pairs):
    a, b = pairs.T
    signal = x[a].astype(float) - x[b]
    power = abs(np.fft.rfft((signal-signal.mean(1,keepdims=True))*np.hanning(480)))**2
    hz = np.fft.rfftfreq(480,1/120)
    def band(lo,hi):
        return np.log(power[:,(hz>=lo)&(hz<hi)].mean(1).clip(1e-20))
    low = band(8,33)
    return low,band(34,45)-low


def matched(low, people, train, query):
    scale = low[train].std(0)
    assert (scale>0).all()
    low = low / scale
    donor = np.empty(len(query),int)
    dist = np.empty(len(query))
    for person in np.unique(people[query]):
        where = people[query]==person
        bank = train[people[train]!=person]
        dist[where], idx = cKDTree(low[bank]).query(low[query[where]])
        donor[where] = bank[idx]
    assert np.all(people[donor]!=people[query]) and np.all(np.isin(donor,train))
    return donor,dist


def model(low,ratio,duration,train,weights):
    """Columns follow §407's band, pair, statistic order."""
    outputs = duration.reshape(len(low),3,12,2)
    coefs = []
    for j in range(12):
        design = np.column_stack((np.ones(len(train)),low[train,j],ratio[train,j]))
        target = outputs[train,:,j,:].reshape(len(train),6)
        coefs.append(np.linalg.lstsq(design*weights[:,None]**.5,target*weights[:,None]**.5,rcond=None)[0])
    return np.stack(coefs).reshape(12,3,3,2)


def effect(coefs, ratio):
    return (coefs[:,2][None,...]*ratio[:,:,None,None]).transpose(0,2,1,3).reshape(len(ratio),72)


def expected(coefs,low):
    return (coefs[:,0][None,...]+coefs[:,1][None,...]*low[:,:,None,None]).transpose(0,2,1,3).reshape(len(low),72)
