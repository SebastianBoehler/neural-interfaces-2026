"""Paired endpoint scores and additive partition of the earlier boundary shift."""
import json

import numpy as np

from score import HERE, ORIG, sc, summary_factory
from transport import TRACE


def main():
    assert (TRACE / 'replay.log').read_text().splitlines()[-1] == 'OK'
    with np.load(ORIG / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cell, person = (z[k + '|all_loso'] for k in ('y', 'cell', 'person'))
    counts = np.array([len(np.unique(cell[person == p])) for p in sc.PEOPLE])
    summary = summary_factory(counts)
    names = ['control', 'RB', 'tRB', 'tL', 'tD']
    scores = {k: np.empty((8, 10)) for k in names}
    parts = {k: np.empty((8, 10)) for k in ('flipped', 'stable')}
    for i, o in enumerate(sc.ORDERS):
        for name in names:
            fname = (HERE / f'final_{name}_{o}_B2_pool.npz' if name.startswith('t') else
                     ORIG / f'runs/s281_v66_cp/{name}_{o}_B2_pool.npz')
            with np.load(fname) as z:
                scores[name][i] = sc.person_accuracy(z['scores'], y, cell, person, sc.PEOPLE)
        with np.load(TRACE / f'RB_{o}.npz') as n, np.load(TRACE / f'L_{o}.npz') as learned:
            p, r = [n[f'B2_pool|{k}'] for k in ('mix', 'rebal')]
            q = learned['B2_pool|mix']
            pred = [a.argmax(1) for a in (p, r, q, q * (r / p))]
            effect = sum(sign * (a == y).astype(float) for sign, a in zip((1,-1,-1,1), pred, strict=True))
            stable = pred[0] == pred[1]
            for name, mask in [('stable', stable), ('flipped', ~stable)]:
                delta = effect * mask
                for j, who in enumerate(sc.PEOPLE):
                    parts[name][i,j] = 100 * np.mean([
                        np.mean([delta[(cell == c) & (y == k)].mean() for k in np.unique(y[cell == c])])
                        for c in np.unique(cell[person == who])])
    assert np.array_equal(scores['control'], scores['tRB'])
    result = {k: summary(v) for k,v in scores.items()}
    for a,b in [('tL','control'), ('tL','tD'), ('tL','RB'), ('tD','control')]:
        result[a+'-'+b] = summary(scores[a]-scores[b])
    result['partition'] = {k: summary(v) for k,v in parts.items()}
    result['source_sha256'] = (HERE/'source.sha256').read_text()
    assert abs(sum(x['mean'] for x in result['partition'].values()) - (-0.631)) < 0.004
    (HERE/'result.json').write_text(json.dumps(result,indent=1))


if __name__ == '__main__':
    main()
