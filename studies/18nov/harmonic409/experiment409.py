"""Source fits, independent interface transfers, and complete-system actor for §409."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from harmonics import features

BASE = Path('/mnt/lustre/work/gehler/gfh527')
FIT = BASE/'ni26_burst407_loo_v2_20261118'
MOTOR = BASE/'ni26_burst407_motor_v1_20261118'
for path, digest in ((FIT/'experiment407.py', 'd36f85a1636bec45aae041a4a3e726511a486899e49012e3f6bd478a3eb6ae6c'),
                     (MOTOR/'motor407.py', '7f21d26facc5aea6bad6bfa8a8967f1303ba03a8aee015e0390d5f856883aaca')):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
sys.path[:0] = [str(FIT), str(MOTOR)]
import experiment407 as owner
import motor407 as actor

owner.features = features


def all_classes(raw, member):
    mass = raw.sum(1, keepdims=True)
    out = .9*raw + .1*mass*member
    np.testing.assert_allclose(out.sum(1), mass[:, 0], rtol=0, atol=1e-14)
    assert np.min(out) >= 0 and np.max(abs(out-raw)) > .01
    return out


def main(mode):
    if mode in ('actor-smoke', 'order', 'score'):
        actor.FIT = Path.cwd()
        actor.motor_update = all_classes
        actor.main(mode)
        return
    if mode in ('fit', 'source', 'extract'):
        owner.main(mode)
        return
    assets, _, _ = owner.source.setup()
    path = assets.CHAIN/'data/proteus_windows.npz'
    with np.load(path) as z:
        people = np.array([p.split('/')[-1] for p in z['subject']])
        train = np.flatnonzero(~np.isin(people, owner.source.HELD))
        if mode == 'smoke':
            names = z['ch_names'].tolist()
            pairs = np.array([(names.index(a), names.index(b)) for a,b in owner.parent.PAIRS])
            row = z['X'][train[0]]
            probe = features(row, pairs, True)
            for left, right in zip(probe, features(-row, pairs)):
                np.testing.assert_allclose(left, right, atol=1e-12)
            assert all(np.isfinite(v).all() for v in probe)
            assert all(np.max(abs(probe[1]-v)) > .01 for v in probe[2:])
            return
        assert mode == 'cross'
        meta = {k:z[k][train] for k in ('subject','session','interface')}
    people = people[train]
    interface = meta['interface']
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'), meta['session'])
    with np.load(path) as z:
        y = owner.source.read_training_labels(path, train, len(z['subject'])).astype(int)
    with np.load('fields.npz') as z:
        matrices = [z[k][train] for k in z.files]
    from run import owner as scoring_owner
    _, metric = scoring_owner()
    report = {}
    for iface in ('Graz','BrainHero'):
        report[iface] = {}
        for person in np.unique(people):
            keep = np.flatnonzero((people != person) & (interface != iface))
            mask = (people == person) & (interface == iface)
            test = np.flatnonzero(mask)
            assert len(np.unique(people[keep])) == 5 and len(test)
            pred = owner.predict(matrices, keep, test, y[keep], meta['stream'], interface)
            report[iface][person] = {
                k:100*metric(y[test], v.argmax(1), meta, mask)['balanced_accuracy_cell_mean']
                for k,v in pred.items()}
    Path('cross.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
