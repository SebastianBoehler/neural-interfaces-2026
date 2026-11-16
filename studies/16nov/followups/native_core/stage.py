"""Stage only a complete source-locked native-delta matrix into a private chain."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from member_chain import ARMS, ORDERS, validate_delta  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    source = Path(os.environ['S301_PRODUCER_ROOT']).resolve(strict=True)
    expected = os.environ['S301_PRODUCER_SHA']
    if sha(source / 'producer.sha256') != expected:
        raise ValueError('producer source lock differs')
    manifest = source / 'runs/s301_native/manifest.json'
    lock = json.loads(manifest.read_text())
    if (lock['producer_source_sha256'] != expected or len(lock['shards']) != 10 or
            set(lock['files']) != {f'{a}_{o}.npz' for a in ARMS for o in ORDERS}):
        raise ValueError('incomplete native producer matrix')
    for name, digest in lock['shards'].items():
        if sha(source / 'runs/s301_native_shards' / name) != digest:
            raise ValueError(f'changed fold shard {name}')
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        orders = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    target = ROOT / 'runs/s301_native/deltas'
    out = target.parent / 'manifest.json'
    if target.exists() or out.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    files = {}
    for name, digest in sorted(lock['files'].items()):
        arm, order = name[:-4].split('_', 1)
        src, dst = source / 'runs/s301_native/deltas' / name, target / name
        if sha(src) != digest:
            raise ValueError(f'changed member delta {name}')
        with np.load(src, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            order_sha = hashlib.sha256(orders[order].tobytes()).hexdigest()
            if (meta['arm'], meta['order'], meta['order_sha256'], meta['producer_source_sha256']) != (arm, order, order_sha, expected):
                raise ValueError(f'delta identity/order mismatch: {name}')
            validate_delta(z['delta'], 2655, strict_zero=arm == 'Rzero')
        with src.open('rb') as f, dst.open('xb') as g:
            shutil.copyfileobj(f, g)
        if sha(src) != digest or sha(dst) != digest:
            raise ValueError(f'delta changed during copy: {name}')
        files[name] = digest
    with out.open('x') as f:
        json.dump({'files': files, 'producer_source_sha256': expected, 'producer_manifest_sha256': sha(manifest)}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'24 locked native deltas staged: {sha(out)}; no score', flush=True)


if __name__ == '__main__':
    main()
