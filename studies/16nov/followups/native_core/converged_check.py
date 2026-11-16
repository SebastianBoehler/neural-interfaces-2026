"""Guard the completed real §305 P01 smoke before the remaining folds."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

producer = Path(os.environ['S305_PRODUCER']).resolve(strict=True)
target = Path(os.environ['S305_SMOKE_GATE'])
path = producer / 'runs/s305_converged_shards/P01.npz'
orders = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
if target.exists():
    raise FileExistsError(target)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = sha(path)
with np.load(path, allow_pickle=False) as z:
    m = json.loads(str(z['metadata'].item())); rows = z['rows']
    if (m['held'] != 'P01' or m['held_accuracy_computed'] is not False or
            m['converged_source_sha256'] != sha(producer / 'converged.sha256') or
            abs(m['native_fit_loss'] - m['strict_loss']) > 1e-5 or
            max(m['max_gradient'], m['strict_max_gradient']) > 1e-5 or
            m['objective_evaluations'] <= 0 or m['strict_iterations'] <= 0 or
            set(m['strict_vs_native_by_order']) != set(orders)):
        raise ValueError('P01 convergence smoke guard failed')
    for name, shape in (('Nconverged_coef', (3, 44239)), ('Rconverged_coef', (3, 44239)),
                        ('Nconverged_intercept', (3,)), ('Rconverged_intercept', (3,))):
        if z[name].shape != shape or not np.isfinite(z[name]).all():
            raise ValueError(f'invalid fitted head {name}')
    for order in orders:
        gaps = m['strict_vs_native_by_order'][order]
        if not np.isfinite([gaps['probability_max_abs'], gaps['logit_max_abs']]).all():
            raise ValueError('nonfinite solver comparison')
        for arm in ('Nfixed', 'Rconverged', 'Nconverged'):
            delta = z[f'{arm}|{order}']
            if delta.shape != (len(rows), 3) or not np.isfinite(delta).all() or not np.allclose(delta.sum(1), 0, atol=1e-10, rtol=0):
                raise ValueError(f'invalid delta: {arm}/{order}')
            if arm == 'Nfixed' and np.max(np.abs(delta), initial=0) > 1e-5:
                raise ValueError(f'fixed native member exceeds source-parity tolerance: {order}')
if sha(path) != before:
    raise ValueError('P01 smoke changed during validation')
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('x') as f:
    json.dump({'artifact_sha256': before, 'converged_source_sha256': sha(producer / 'converged.sha256'),
               'held_accuracy_computed': False}, f, sort_keys=True); f.write('\n')
print('P01 converged smoke passed; no score', flush=True)
