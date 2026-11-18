"""Change which historic rows enter CP while preserving every class quota."""
import numpy as np

CORE = ('Z','B','P386','F386','P1386','F1386')
GEOM = ('G','H')


def exchange(y, c, original, arm, feature):
    if arm == 'Z':
        return original
    chosen = np.zeros(len(y), dtype=bool)
    chosen[original] = True
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        kept = rows[chosen[rows]]
        omitted = rows[~chosen[rows]]
        count = min(len(omitted), int(len(kept)/8+.5))
        if not count:
            continue
        if arm == 'B':
            priority = c
        elif arm in GEOM:
            x = feature[rows]
            if arm == 'H':
                rng = np.random.default_rng([2386, int(label), len(y)])
                x = x[rng.permutation(len(x))]
            x = (x-x.mean(0))/np.maximum(x.std(0), 1e-5)
            priority = np.zeros(len(y))
            priority[rows] = x @ (-x[chosen[rows]].mean(0))
        else:
            prefix = len(y) if arm[0] == 'F' else 0
            rng = np.random.default_rng([int(arm[1:]), int(label), prefix])
            priority = rng.random(len(y))
        remove = kept[np.argsort(priority[kept], kind='stable')[:count]]
        add = omitted[np.argsort(-priority[omitted], kind='stable')[:count]]
        chosen[remove],chosen[add] = False,True
    return np.flatnonzero(chosen)
