"""Inject a checked §295 core posterior delta into the existing v6 outer chain."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ARMS = ('geo_zero', 'geo_s50', 'geo_d25', 'geo_d50', 'geo_d100')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def validate_delta(delta, n, zero):
    a = np.asarray(delta)
    if (a.shape != (n, 3) or not np.isfinite(a).all() or np.max(np.abs(a), initial=0) > 1 or
            not np.allclose(a.sum(1), 0, atol=1e-10, rtol=0)):
        raise ValueError('invalid geometry posterior delta')
    if (zero and np.any(a != 0)) or (not zero and not np.any(a != 0)):
        raise ValueError('zero or active geometry delta is a no-op')


def patch_chain(source):
    anchor = '    HB = {"B2_pool":'
    if source.count(anchor) != 1:
        raise ValueError('expected one chain insertion anchor')
    inject = '''    if arm == "V6Fr":
        with np.load(os.environ["S295_DELTA"]) as gz:
            GEO_DELTA = gz["delta"]
        if np.any(GEO_DELTA):
            A.C1[p0] = A.C1[p0] + GEO_DELTA[P]
            Z[f"{bn}|Ps|{p0}"][0] = Z[f"{bn}|Ps|{p0}"][0] + GEO_DELTA[P] / 2
'''
    return source.replace(anchor, inject + anchor, 1)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(arm, order):
    if arm not in ARMS or order not in ORDERS:
        raise ValueError('unknown registered geometry arm/order')
    root = Path(os.environ['NI26_WS']).resolve(strict=True)
    manifest = Path(os.environ['S295_MANIFEST']).resolve(strict=True)
    if sha(manifest) != os.environ['S295_MANIFEST_SHA']:
        raise ValueError('geometry manifest lock mismatch')
    lock = json.loads(manifest.read_text())
    name = f'{arm}_{order}.npz'
    path = root / 'runs/s295_geometry/deltas' / name
    if name not in lock['files'] or sha(path) != lock['files'][name]:
        raise ValueError('geometry delta absent or hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata'].item()))
        if (metadata.get('arm'), metadata.get('order')) != (arm, order):
            raise ValueError('geometry delta identity mismatch')
        validate_delta(z['delta'], 2655, zero=arm == 'geo_zero')
    base_path = root / 'studies/14nov/newmember/run_s_v66.py'
    spec = importlib.util.spec_from_file_location('s295_base', base_path)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    original = base.patch_chain
    base.patch_chain = lambda src: patch_chain(original(src))

    def recipe(name, add, swap, zero, seeds):
        if name != arm or add != 'none' or swap != 'none' or zero != '0' or seeds != '0,1,2,3,4,5':
            raise ValueError('invalid §295 chain configuration')
        return {'name': name, 'add': add, 'swap': swap, 'zero': name == 'geo_zero', 'seeds': seeds}
    base.validate_recipe = recipe
    original_tracker = base.InputTracker

    class Tracker(original_tracker):
        def __init__(self):
            super().__init__()
            self.track(Path(__file__).resolve())
            self.track(manifest)
    base.InputTracker = Tracker
    os.environ.update(NM_NAME=arm, ADD='none', SWAP='none', ZERO='0', NM_SEEDS='0,1,2,3,4,5',
                      S295_DELTA=str(path))
    sys.argv = [str(base_path), order]
    base.main()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: chain.py ARM ORDER')
    main(*sys.argv[1:])
