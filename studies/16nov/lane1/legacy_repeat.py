"""Measure repeated historical one-fold fitting variability, without scoring."""
import importlib.util
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve()
SPEC = importlib.util.spec_from_file_location('multistream', ROOT / 'studies/16nov/lane1/multistream.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def main(family):
    if family not in ('eegn', 'eegn_dist2'):
        raise ValueError('expected one of the two EEGNet families')
    t7 = M.load_module(ROOT); d = t7.protocol.load()
    t7.TAG = 'ttt18within0'; t7.PERM = {'all_loso': M.permutations(ROOT, len(d['y']))['within0']}
    paths = []
    for repeat in range(2):
        out = ROOT / f'runs/diag/ttt/lane1_legacy_repeat{repeat}'
        out.mkdir(parents=True, exist_ok=True)
        path = out / f'ttt18within0_{family}_base-all_loso-s0.npz'
        if path.exists():
            raise FileExistsError(path)
        t7.run(family, d, 'all_loso', 0, out, epochs=1); paths.append(path)
    with np.load(paths[0]) as a, np.load(paths[1]) as b:
        if not np.array_equal(a['index'], b['index']):
            raise ValueError('repeat indices differ')
        delta = np.abs(a['proba'] - b['proba'])
        labels = np.count_nonzero(a['proba'].argmax(1) != b['proba'].argmax(1))
        print(f'legacy repeat {family}: maxdiff={delta.max():.9g} mean={delta.mean():.9g} labels={labels}', flush=True)


if __name__ == '__main__':
    main(('eegn', 'eegn_dist2')[int(sys.argv[1])])
