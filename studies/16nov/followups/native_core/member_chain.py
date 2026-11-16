"""Inject a validated native core delta into the existing §295 full chain."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ARMS = ('Rzero', 'Nfixed', 'Ntrain')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def validate_delta(delta, n, strict_zero):
    a = np.asarray(delta)
    if (a.shape != (n, 3) or not np.isfinite(a).all() or np.max(np.abs(a), initial=0) > 1 or
            not np.allclose(a.sum(1), 0, atol=1e-10, rtol=0) or (strict_zero and np.any(a != 0))):
        raise ValueError('invalid native member delta or strict zero')


def verify_arm(control, arm, strict_zero):
    if set(control) != set(arm):
        raise ValueError('gate record keys differ')
    bases = {k.rsplit('|', 1)[0] for k in control if k.endswith('|V6Fr')}
    if not bases or set(control) != {f'{b}|{k}' for b in bases for k in ('input', 'gate', 'V6Fr')}:
        raise ValueError('incomplete gate records')
    changed = 0
    for base in bases:
        for k in ('input', 'gate'):
            if not np.array_equal(control[f'{base}|{k}'], arm[f'{base}|{k}']):
                raise ValueError(f'{base}: inner gate differs')
        same = np.array_equal(control[f'{base}|V6Fr'], arm[f'{base}|V6Fr'])
        if strict_zero and not same:
            raise ValueError(f'{base}: strict zero changed output')
        changed += not same
    print(f'inner gates unchanged; final-output bases changed={changed}/{len(bases)}', flush=True)


def main(arm, order):
    if arm not in ARMS or order not in ORDERS:
        raise ValueError('unknown native core arm or order')
    root = Path(os.environ['NI26_WS']).resolve(strict=True)
    sys.path.insert(0, str(root / 'studies/16nov/followups/geometry'))
    from chain import patch_chain, sha  # noqa: E402
    manifest = Path(os.environ['S301_MANIFEST']).resolve(strict=True)
    if sha(manifest) != os.environ['S301_MANIFEST_SHA']:
        raise ValueError('native delta manifest lock mismatch')
    lock = json.loads(manifest.read_text())
    name = f'{arm}_{order}.npz'
    path = root / 'runs/s301_native/deltas' / name
    if set(lock['files']) != {f'{a}_{o}.npz' for a in ARMS for o in ORDERS} or sha(path) != lock['files'][name]:
        raise ValueError('native delta matrix incomplete or checksum differs')
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata'].item()))
        if (metadata.get('arm'), metadata.get('order')) != (arm, order):
            raise ValueError('native delta identity differs')
        validate_delta(z['delta'], 2655, strict_zero=arm == 'Rzero')
    base_path = root / 'studies/14nov/newmember/run_s_v66.py'
    spec = importlib.util.spec_from_file_location('s301_existing_base', base_path)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    original = base.patch_chain
    base.patch_chain = lambda source: patch_chain(original(source))
    base.verify = lambda control, values, zero: verify_arm(control, values, strict_zero=zero)

    def recipe(name, add, swap, zero, seeds):
        if name != arm or add != 'none' or swap != 'none' or zero != '0' or seeds != '0,1,2,3,4,5':
            raise ValueError('invalid native member chain configuration')
        return {'name': name, 'add': add, 'swap': swap, 'zero': name == 'Rzero', 'seeds': seeds}
    base.validate_recipe = recipe
    tracker = base.InputTracker

    class Tracker(tracker):
        def __init__(self):
            super().__init__()
            self.track(Path(__file__).resolve())
            self.track(root / 'studies/16nov/followups/geometry/chain.py')
            self.track(manifest)
    base.InputTracker = Tracker
    os.environ.update(NM_NAME=arm, ADD='none', SWAP='none', ZERO='0', NM_SEEDS='0,1,2,3,4,5', S295_DELTA=str(path))
    sys.argv = [str(base_path), order]
    base.main()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: member_chain.py ARM ORDER')
    main(*sys.argv[1:])
