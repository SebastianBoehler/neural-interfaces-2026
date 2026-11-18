"""Fixed-quota confidence-band interventions for the pinned §386 replay worker."""
import numpy as np

CORE = ('Z','N386','N1386','A386','A1386')
GEOM = ('G','H','R')


def exchange(y,c,original,arm,feature):
    if arm=='Z':
        return original
    chosen = np.zeros(len(y),dtype=bool)
    chosen[original] = True
    for label in np.unique(y):
        rows = np.flatnonzero(y==label)
        kept = rows[chosen[rows]]
        out = rows[~chosen[rows]]
        n = min(len(out),int(len(kept)/8+.5))
        if not n:
            continue
        # Rank each side from nearest to farthest from its class's cutoff.
        drop = kept[np.argsort(c[kept],kind='stable')]
        add = out[np.argsort(-c[out],kind='stable')]
        if arm.startswith('A'):
            drop,add = drop[::-1],add[::-1]
        drop,add = drop[:2*n],add[:2*n]
        if arm in GEOM:
            if arm=='R':
                score = np.random.default_rng([390,int(label),len(y)]).random(len(y))
            else:
                x = feature[rows]
                if arm=='H':
                    rng = np.random.default_rng([2390,int(label),len(y)])
                    x = x[rng.permutation(len(rows))]
                x = (x-x.mean(0))/np.maximum(x.std(0),1e-5)
                score = np.zeros(len(y))
                score[rows] = x @ (-x[chosen[rows]].mean(0))
            remove = [part[np.argmin(score[part])] for part in np.array_split(drop,n)]
            include = [part[np.argmax(score[part])] for part in np.array_split(add,n)]
        else:
            seed = int(arm[1:])
            score = np.random.default_rng([seed,int(label),len(y)]).random(len(y))
            remove = drop[np.argsort(score[drop],kind='stable')[:n]]
            include = add[np.argsort(-score[add],kind='stable')[:n]]
        chosen[remove],chosen[include] = False,True
    return np.flatnonzero(chosen)
