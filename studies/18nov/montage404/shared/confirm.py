"""Confirm learned cross-interface geometry with all transformations excluding one person."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
PRIOR = ROOT / 'ni26_align404b_v1_20261118'
sys.path.insert(0, str(PRIOR))
import fit
import fit_shared


def run(start, stop):
    fit.check(fit.digest(PRIOR/'source.sha256') == '9b4f9cf79557295992a385576e9eea9797e6a394a29753edbc8d70f9bea85db8', 'parent')
    for line in (PRIOR/'source.sha256').read_text().splitlines():
        h, name = line.split(maxsplit=1)
        fit.check(fit.digest(PRIOR/name) == h, 'parent source lock')
    assets, _, _ = fit.setup()
    path = assets.CHAIN/'data/proteus_windows.npz'
    with np.load(path) as z:
        person = np.array([p.rsplit('/', 1)[-1] for p in z['subject']])
        rows = np.flatnonzero(~np.isin(person, fit.HELD))
        session = z['session'][rows]
        src = fit.features(z['X'][rows])
    iface = fit.read_selected(path, 'interface', rows, len(person))
    y = fit.read_selected(path, 'y', rows, len(person))
    people = person[rows]
    unique = np.unique(people)
    fit.check(len(unique) == 6 and len(rows) == 1898, 'source ownership')
    results = {}
    for i in range(start, stop):
        p = unique[i]
        take = people != p
        train = src[take]
        mean, std = train.mean(0), train.std(0).clip(1e-6)
        train = (train-mean)/std
        _, _, basis = randomized_svd(train, n_components=96, random_state=404, n_iter=6)
        training_pc = train @ basis.T
        vals, u = np.linalg.eigh(training_pc.T @ training_pc/(len(train)-1))
        fit.check(vals.min() > 1e-8, 'source whitening')
        low = (((src-mean)/std) @ basis.T) @ (u/np.sqrt(vals))
        fit.check(np.max(abs(np.cov(low[take].T)-np.eye(96))) < 1e-7, 'fold whitening')
        a, b = fit.centroids(low, people, iface, session, y)
        results[str(p)] = dict(training=unique[unique != p].tolist(),
                               difference=fit.agreement(a, b, i),
                               shared=fit_shared.agreement(a, b, i))
    Path(f'folds_{start}.json').write_text(json.dumps(dict(source=fit.digest('source.sha256'),results=results), indent=2))


if __name__ == '__main__':
    run(*map(int, sys.argv[1:]))
