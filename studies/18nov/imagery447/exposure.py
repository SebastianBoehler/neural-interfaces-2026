"""Identify previously exported people by original event times, without EEG decoding."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat


def verify(repo, people):
    with np.load(repo / 'data/chang2025/windows_1-28.npz') as previous:
        p, s, t = (previous[k] for k in ('subject', 'session', 'onset'))
    signatures = {}
    for person, session in set(zip(p, s)):
        onsets = tuple(np.rint(1000 * t[(p == person) & (s == session)]).astype(int))
        assert onsets not in signatures
        signatures[onsets] = str(person)
    identified = defaultdict(set)
    matches = set()
    recordings = repo / 'data/moabb_raw/MNE-chang2025-data'
    for path in recordings.glob('S2*/S2*_MI*.set'):
        events = loadmat(path, simplify_cells=True)['EEG']['event']
        sig = tuple(int(e['latency']) - 1 for e in events if str(e['type']) in ('1','2','3'))
        if sig in signatures:
            identified[path.parent.name].add(signatures[sig])
            matches.add(sig)
    assert len(matches) == len(signatures) == 89 and len(identified) == 22
    assert all(len(labels) == 1 for labels in identified.values())
    assert set(identified).isdisjoint(f'S{i}' for i in people)
    return dict(sessions=len(matches), people={k: next(iter(v)) for k, v in sorted(identified.items())})


if __name__ == '__main__':
    print(verify(Path(sys.argv[1]), (211,216,221,222,226,227)))
