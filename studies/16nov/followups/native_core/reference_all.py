"""Export ten training-only sklearn scaler/head references, without scoring."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
sys.path.insert(0, str(ROOT / 'experiments'))
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    output = ROOT / 'runs/s301_reference_all'
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    d = protocol.load()
    fe = T.Feats(d, T.parse('car_ov17_fs2_t0.1_le_k2'))
    if fe.F.shape[1] != 44239 or len(fe.C) != 51:
        raise ValueError('unexpected reference features')
    seen = set()
    for held, tr, te in protocol.folds(d, 'all_loso'):
        fit = np.flatnonzero(tr)
        if held in seen or len(np.unique(d['subject'][tr])) != 9 or set(d['subject'][tr]) & set(d['subject'][te]):
            raise ValueError('held-person fold is invalid')
        seen.add(held)
        model = T.fit_head('lr0.3', fe.F[fit], d['y'][fit], fe.slices)
        scaler, head = model.steps[0][1], model.steps[1][1]
        if not np.array_equal(head.classes_, [0, 1, 2]):
            raise ValueError('head class order changed')
        path = output / f'{held}.npz'
        with path.open('xb') as f:
            np.savez(f, fit_rows=fit.astype(np.int64), mean=scaler.mean_, scale=scaler.scale_,
                     coef=head.coef_, intercept=head.intercept_)
        print(f'{held} training-only reference head exported; no score', flush=True)
    if seen != {f'P{i:02d}' for i in range(1, 11)}:
        raise ValueError('ten complete folds required')
    record = {'source_sha256': sha(ROOT / 'source.sha256'), 'data_sha256': sha(ROOT / 'data/proteus_windows.npz'),
              'held_accuracy_computed': False, 'files': {p.name: sha(p) for p in sorted(output.glob('*.npz'))}}
    with (output / 'manifest.json').open('x') as f:
        json.dump(record, f, indent=2, sort_keys=True); f.write('\n')


if __name__ == '__main__':
    main()
