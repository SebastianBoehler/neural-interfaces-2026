"""Measure regional burst duration with the immutable §402 readout and §401 actor."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler
from bursts import features

BASE = Path('/mnt/lustre/work/gehler/gfh527')
PARENT = BASE/'ni26_phase402_v1_20261118'
for line in (PARENT/'source.sha256').read_text().splitlines():
    expected, name = line.split()
    assert hashlib.sha256((PARENT/name).read_bytes()).hexdigest() == expected
assert hashlib.sha256((PARENT/'experiment402.py').read_bytes()).hexdigest() == '240750e8b4df9ec5119f98621ba40f43beabcffcfc9ea44b21a78b9483a0889c'
sys.path.insert(0, str(PARENT))
import experiment402 as parent

source = parent.source
ARMS = ('P', 'X', 'D', 'B')


def predict(fields, train, test, labels, stream, interface):
    weights = source.design_weights(stream[train], interface[train], labels)
    results = {}
    for i, name in enumerate(ARMS):
        matrix = fields[0] if i == 0 else np.column_stack((fields[0], fields[i]))
        scaler = StandardScaler().fit(matrix[train], sample_weight=weights)
        a, b = scaler.transform(matrix[train]), scaler.transform(matrix[test])
        distances = pdist(a, 'sqeuclidean')
        width = np.median(distances)
        parent.require(width > 0, 'zero bandwidth')
        k = np.exp(-squareform(distances)/width)
        q = np.exp(-cdist(b, a, 'sqeuclidean')/width)
        results[name] = source.predict(k, q, weights, np.eye(3)[labels])
    for ctrl in ('P', 'D', 'B'):
        parent.require(np.max(abs(results['X'] - results[ctrl])) > .001, 'inactive '+ctrl)
    return results


def main(mode):
    if mode in ('actor-smoke', 'order', 'score'):
        parent.ARMS = ARMS
        parent.main()
        return
    assets, _, _ = source.setup()
    path = assets.CHAIN/'data/proteus_windows.npz'
    with np.load(path) as data:
        people = np.array([p.split('/')[-1] for p in data['subject']])
        train = np.flatnonzero(~np.isin(people, source.HELD))
        target = np.flatnonzero(np.isin(people, source.HELD))
        stream = np.char.add(np.char.add(data['subject'], '|'), data['session'])
        interface = data['interface']
        if mode in ('smoke', 'extract'):
            x, names = data['X'], data['ch_names'].tolist()
            pairs = np.array([(names.index(a), names.index(b)) for a,b in parent.PAIRS])
            if mode == 'smoke':
                probe = features(x[train[0]], pairs, True)
                for v, w in zip(probe, features(-x[train[0]], pairs)):
                    np.testing.assert_array_equal(v,w)
                assert all(np.max(abs(probe[1]-v)) > .01 for v in probe[2:])
                return
            np.savez('fields.npz', *map(np.stack, zip(*(features(row, pairs) for row in x))))
            return
    with np.load('fields.npz') as f:
        fields = [f[k] for k in f.files]
    labels = source.read_training_labels(path, train, len(people)).astype(int)
    if mode == 'source':
        from run import owner
        _, official = owner()
        with np.load(path) as z:
            meta = {k:z[k][train] for k in ('subject','session','interface')}
        meta['stream'] = stream[train]
        local = [f[train] for f in fields]
        result = {}
        for p in np.unique(people[train]):
            mask = people[train] == p
            fit, test = np.flatnonzero(~mask), np.flatnonzero(mask)
            probs = predict(local, fit, test, labels[fit], stream[train], interface[train])
            result[p] = {a:100*official(labels[test], v.argmax(1), meta, mask)['balanced_accuracy_cell_mean']
                         for a,v in probs.items()}
        Path('loo.json').write_text(json.dumps(result,indent=2))
        return
    parent.require(mode == 'fit', 'unknown mode')
    result = predict(fields, train, target, labels, stream, interface)
    np.savez_compressed('members.npz', index=target, **result)
    card = dict(source=parent.digest('source.sha256'), members=parent.digest('members.npz'),
                held=source.HELD, train_people=sorted(set(people[train])), query_labels_opened=False)
    Path('source.json').write_text(json.dumps(card, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
