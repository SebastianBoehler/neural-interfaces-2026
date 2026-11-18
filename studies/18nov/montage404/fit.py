"""Test a source-person-paired interface metric using only official EEG at inference."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.utils.extmath import randomized_svd

ROOT = Path('/mnt/lustre/work/gehler/gfh527')
PRIOR = ROOT / 'ni26_priv401_source_v1_20261118'
sys.path.insert(0, str(PRIOR))
from source import ARMS, HELD, check, design_weights, digest, features, predict, setup
from source_labels import read_selected


def axes(graz, brain, shift):
    delta = graz - np.roll(brain, shift, axis=0)
    _, s, vt = np.linalg.svd(delta.reshape(-1, delta.shape[-1]), full_matrices=False)
    check(s[2] > 1e-8, 'insufficient interface directions')
    return vt[:3].T, s[:3].tolist()


def centroids(src, person, iface, session, y):
    people = np.unique(person)
    cells = np.empty((2, len(people), 3, src.shape[1]))
    for i, f in enumerate(('Graz', 'BrainHero')):
        for j, p in enumerate(people):
            for c in range(3):
                mask = (person == p) & (iface == f) & (session == '1') & (y == c)
                check(mask.sum() >= 2, 'missing matched source cells')
                cells[i, j, c] = src[mask].mean(0)
    cells -= cells.mean(2, keepdims=True)
    return cells


def fraction(a, v):
    return float(np.square(a @ v).sum() / np.square(a).sum())


def main(smoke):
    check(digest(PRIOR / 'source.sha256') == '587b2cae97ec18e463c17676ccc79ae4ec7178b60f91977087e68beef9c8de64', 'source imports')
    for line in (PRIOR / 'source.sha256').read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        check(digest(PRIOR / name) == sha, 'changed prior source')
    assets, _, _ = setup()
    path = assets.CHAIN / 'data/proteus_windows.npz'
    with np.load(path) as z:
        person = np.array([p.rsplit('/', 1)[-1] for p in z['subject']])
        rows = np.flatnonzero(~np.isin(person, HELD))
        held = np.flatnonzero(np.isin(person, HELD))
        x = z['X']
        session = z['session'][rows]
    iface = read_selected(path, 'interface', rows, len(person))
    y = read_selected(path, 'y', rows, len(person))
    people = person[rows]
    check(set(people) == {'P04','P05','P06','P07','P08','P10'}, 'source people')
    if smoke:
        take = np.zeros(len(rows), dtype=bool)
        for p in np.unique(people):
            for f in np.unique(iface):
                for c in range(3):
                    ids = np.flatnonzero((session == '1') & (people == p) & (iface == f) & (y == c))
                    take[ids[:4]] = True
        rows, session, iface, y, people = (a[take] for a in (rows, session, iface, y, people))
    else:
        check(json.loads(Path('fit_smoke.json').read_text())['source'] == digest('source.sha256'), 'source smoke')
    src = features(x[rows])
    query = features(x[rows[:8] if smoke else held])
    mean, std = src.mean(0), np.maximum(src.std(0), 1e-6)
    src = (src - mean) / std
    _, _, basis = randomized_svd(src, n_components=96, random_state=404, n_iter=6)
    low = src @ basis.T
    ev, v = np.linalg.eigh(low.T @ low / (len(rows)-1))
    check(ev.min() > 1e-8, 'source eigenvalues')
    white = v / np.sqrt(ev)
    src = low @ white
    query = ((query-mean)/std) @ basis.T @ white
    check(np.max(abs(np.cov(src.T)-np.eye(96))) < 1e-7, 'source whitening')
    graz, brain = centroids(src, people, iface, session, y)
    directions, spectra = {}, {}
    for a, offset in [('X',0),('D1',1),('D2',3)]:
        directions[a], spectra[a] = axes(graz, brain, offset)
    directions['M'] = directions['X']
    audit = []
    for j in range(len(graz)):
        keep = np.arange(len(graz)) != j
        delta = graz[j] - brain[j]
        frac = [fraction(delta, axes(graz[keep], brain[keep], shift)[0]) for shift in range(5)]
        audit.append(frac[0] - np.mean(frac[1:]))
    base = squareform(pdist(src, 'sqeuclidean'))
    held_dist = cdist(query, src, 'sqeuclidean')
    scale = np.median(base[np.triu_indices(len(src), 1)])
    weights = design_weights(np.char.add(people, session), iface, y)
    outputs = {}
    for a in ARMS:
        train, test = base, held_dist
        if a != 'O':
            proj = src @ directions[a]
            check(abs(np.var(proj, axis=0, ddof=1).sum()-3) < 1e-7, 'projection energy')
            factor = .75 if a == 'M' else -.75
            train = base + factor * squareform(pdist(proj, 'sqeuclidean'))
            test = held_dist + factor * cdist(query @ directions[a], proj, 'sqeuclidean')
        outputs[a] = predict(np.exp(-train/scale), np.exp(-test/scale), weights, np.eye(3)[y])
    check(all(np.max(abs(outputs['X']-outputs[a])) > 1e-6 for a in ARMS if a != 'X'), 'inactive arms')
    card = dict(source=digest('source.sha256'), query_labels_opened=False, eligible=len(rows),
                held=len(held), spectra=spectra, source_people=sorted(set(people)), source_audit=audit)
    if not smoke:
        np.savez_compressed('members.npz', index=held, **outputs)
        card['members'] = digest('members.npz')
    Path('fit_smoke.json' if smoke else 'source.json').write_text(json.dumps(card, indent=2))


if __name__ == '__main__':
    main(sys.argv[1] == 'smoke')
