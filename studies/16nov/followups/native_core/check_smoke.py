"""Validate real P01 preparation or fit artifacts before the remaining wave."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(kind):
    if kind not in ('prepare', 'fit'):
        raise ValueError('expected prepare or fit')
    path = ROOT / ('runs/s301_native_inputs/P01.npz' if kind == 'prepare' else 'runs/s301_native_shards/P01.npz')
    gate = ROOT / f'runs/s301_gates/{kind}_P01.json'
    if gate.exists():
        raise FileExistsError(gate)
    before = sha(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'].item()))
        rows = z['rows']
        if (not len(rows) or meta.get('held_accuracy_computed') is not False or
                meta.get('source_sha256') != sha(ROOT / 'producer.sha256')):
            raise ValueError('smoke provenance failed')
        if kind == 'prepare':
            if (meta['person'] != 'P01' or meta['rows_sha256'] != hashlib.sha256(rows.tobytes()).hexdigest() or
                    z['cov'].shape != (len(rows), 51, 41, 41) or z['slow'].shape != (len(rows), 328) or
                    z['features'].shape != (len(rows), 44239) or z['prior'].shape != (51, 41, 41)):
                raise ValueError('preparation output shape/identity differs')
            for key in ('cov', 'slow', 'features', 'prior'):
                if not np.isfinite(z[key]).all():
                    raise ValueError(f'nonfinite {key}')
        else:
            if (meta['held'] != 'P01' or abs(meta['native_fit_loss'] - meta['reference_loss']) > 1e-5 or
                    meta['max_gradient'] > 1e-5 or meta['objective_evaluations'] <= 0 or
                    set(meta['native_trained_vs_fixed_probability_max_abs']) != set(ORDERS)):
                raise ValueError('fit smoke objective, gradient or order guard failed')
            for order in ORDERS:
                for arm in ('Nfixed', 'Ntrain'):
                    v = z[f'{arm}|{order}']
                    if v.shape != (len(rows), 3) or not np.isfinite(v).all() or not np.allclose(v.sum(1), 0, atol=1e-10, rtol=0):
                        raise ValueError(f'invalid {arm}/{order} delta')
    if sha(path) != before:
        raise ValueError('smoke artifact changed during validation')
    gate.parent.mkdir(parents=True, exist_ok=True)
    with gate.open('x') as f:
        json.dump({'artifact_sha256': before, 'source_sha256': sha(ROOT / 'producer.sha256'),
                   'kind': kind, 'held_accuracy_computed': False}, f, sort_keys=True)
        f.write('\n')
    print(f'{kind} real P01 smoke passed; no score', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: check_smoke.py prepare|fit')
    main(sys.argv[1])
