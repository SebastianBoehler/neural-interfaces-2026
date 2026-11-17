"""Source-only inner-held proxy and matched weight controls."""
import hashlib
import numpy as np
from scipy.special import softmax


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def balanced(group):
    _, inverse, n = np.unique(group, return_inverse=True, return_counts=True)
    return len(group) / (len(n) * n[inverse])


def features(windows):
    freq = np.fft.rfftfreq(480, 1/120)
    bands = [(1,4),(4,8),(8,13),(13,20),(20,30),(30,55)]
    result = []
    for idx in np.array_split(np.arange(len(windows)), (len(windows)+63)//64):
        power = np.abs(np.fft.rfft(windows[idx].astype(np.float64)))**2
        result.append(np.concatenate([np.log(np.maximum(1e-12, power[:,:, (freq>=lo)&(freq<hi)].mean(-1)))
                                      for lo,hi in bands], axis=1))
    return np.concatenate(result)


def proxy_weights(x, y, streams, interfaces, person, rows, outer):
    require(len(set(person)) == 9 and outer not in person, 'invalid teacher outer exclusion')
    group = np.array(['|'.join((s,i,str(c))) for s,i,c in zip(streams,interfaces,y)])
    feat = features(x)
    proba = np.full((len(y),3), np.nan)
    lineage = {}
    for inner in sorted(set(person)):
        fit = person != inner
        source = feat[fit]
        mean, sd = source.mean(0), source.std(0).clip(1e-4)
        z = (source-mean)/sd
        w = balanced(group[fit])
        target = np.eye(3)[y[fit]]
        prior = np.average(target, axis=0, weights=w)
        gram = (z.T*w)@z
        gram.flat[::len(gram)+1] += .1*len(z)
        beta = np.linalg.solve(gram,(z.T*w)@(target-prior))
        proba[~fit] = softmax(((feat[~fit]-mean)/sd)@beta+prior,axis=1)
        lineage[inner] = dict(fit_people=sorted(set(person[fit])), fit_rows=digest(rows[fit]),
                              query_rows=digest(rows[~fit]), fit_features=digest(source))
        require(len(lineage[inner]['fit_people']) == 8, 'inner fit person count')
    require(np.isfinite(proba).all(), 'incomplete teacher bank')
    err = 1 + 2 * (proba.argmax(1) != y)
    uncertainty = 1 - proba.max(1)
    w = {'O':balanced(group)}
    w.update({arm:np.empty(len(y)) for arm in ('E','C','S')})
    rng = np.random.default_rng(376)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        e = err[idx] / err[idx].mean() * w['O'][idx]
        w['E'][idx] = e
        w['C'][idx[np.argsort(uncertainty[idx],kind='stable')]] = np.sort(e)
        w['S'][rng.permutation(idx)] = e
    require(all(np.isclose(a.sum(),len(y)) for a in w.values())
            and len({digest(a) for a in w.values()}) == 4
            and all(np.array_equal(np.sort(w[a]),np.sort(w['E'])) for a in ('S','C')),
            'unmatched/inactive weights')
    return w, dict(inner_folds=lineage, source_only_probabilities_sha=digest(proba),
                   weights={a:digest(v) for a,v in w.items()})
