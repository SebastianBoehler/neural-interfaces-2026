"""Export registered no-score strong-core parity inputs for P01/P10."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
sys.path.insert(0, str(ROOT / 'experiments'))
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402

ORDERS = ('within0', 'full0')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    out = ROOT / 'runs/s301_reference'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    d = protocol.load()
    feats = T.Feats(d, T.parse('car_ov17_fs2_t0.1_le_k2'))
    if feats.F.shape[1] != 44239 or len(feats.C) != 51:
        raise ValueError('reference feature layout changed')
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    n = len(d['subject'])
    if any(not np.array_equal(np.sort(p), np.arange(n)) for p in perms.values()):
        raise ValueError('invalid reference permutation')
    wanted = {'P01', 'P10'}
    folds = {held: (tr, te) for held, tr, te in protocol.folds(d, 'all_loso') if held in wanted}
    if set(folds) != wanted:
        raise ValueError('registered held-person folds are absent')
    for held, (tr, te) in folds.items():
        fit = np.flatnonzero(tr)
        if set(d['subject'][tr]) & set(d['subject'][te]) or len(np.unique(d['subject'][tr])) != 9:
            raise ValueError(f'{held}: held-person exclusion failed')
        priors = np.stack([feats.prior(b, fit) for b in range(51)])
        d2p = [x[fit].mean() for x in feats.d2]
        model = T.fit_head('lr0.3', feats.F[fit], d['y'][fit], feats.slices)
        scaler, classifier = model.steps[0][1], model.steps[1][1]
        if not np.array_equal(classifier.classes_, [0, 1, 2]):
            raise ValueError('head class order changed')
        common = {'prior': priors, 'mean': scaler.mean_, 'scale': scaler.scale_,
                  'coef': classifier.coef_, 'intercept': classifier.intercept_,
                  'fit_rows': fit.astype(np.int64)}
        for order, perm in perms.items():
            rows = perm[te[perm]]
            session = d['stream'][rows[0]]
            rows = rows[d['stream'][rows] == session]
            if len(rows) < 62:
                raise ValueError(f'{held}/{order}: fewer than 62 windows')
            use = rows[:62]
            reference = feats.causal(use, list(priors), d2p, None)
            first = feats.causal(use[:61], list(priors), d2p, None)
            if not np.array_equal(reference[:61], first):
                raise ValueError('reference causal prefix changed with future')
            payload = {**common, 'rows': use, 'raw': d['X'][use], 'expected_features': reference,
                       'expected_probabilities': model.predict_proba(reference)}
            target = out / f'{held}_{order}.npz'
            with target.open('xb') as f:
                np.savez_compressed(f, **payload)
            print(f'{target.name}: {len(use)} rows, no score', flush=True)
    record = {'source_sha256': sha(ROOT / 'source.sha256'), 'data_sha256': sha(ROOT / 'data/proteus_windows.npz'),
              'permutation_sha256': sha(ROOT / 'runs/shuffle_perms.npz'), 'scipy': scipy.__version__,
              'numpy': np.__version__, 'sklearn': sklearn.__version__, 'held_accuracy_computed': False,
              'files': {p.name: sha(p) for p in sorted(out.glob('*.npz'))}}
    with (out / 'manifest.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True); f.write('\n')


if __name__ == '__main__':
    main()
