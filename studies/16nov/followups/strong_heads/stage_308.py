"""Copy only a complete, source-locked §308 delta matrix into its private chain."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'studies/16nov/followups/native_core')]
from definitions import ORDERS, SEED_ARMS  # noqa: E402
from member_chain import validate_delta  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    source = Path(os.environ['S308_OUT']).resolve(strict=True)
    expected = os.environ['S308_PRODUCER_SHA']
    if digest(Path(os.environ['S308_MANIFEST'])) != expected:
        raise ValueError('strong-head producer lock mismatch')
    lock_path = source / 'manifest.json'
    lock = json.loads(lock_path.read_text())
    if (lock['producer_source_sha256'] != expected or len(lock['shards']) != 20 or
            set(lock['files']) != {f'{arm}_{order}.npz' for arm in SEED_ARMS for order in ORDERS}):
        raise ValueError('incomplete strong-head producer matrix')
    for name, sha in lock['shards'].items():
        if digest(source / 'shards' / name) != sha:
            raise ValueError(f'changed source shard: {name}')
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        orders = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    target = ROOT / 'runs/s301_native/deltas'
    output = target.parent / 'manifest.json'
    if target.exists() or output.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    files = {}
    for name, sha in sorted(lock['files'].items()):
        arm, order = name[:-4].split('_', 1)
        src, dst = source / 'deltas' / name, target / name
        if digest(src) != sha:
            raise ValueError(f'changed source delta: {name}')
        with np.load(src, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            order_hash = hashlib.sha256(orders[order].tobytes()).hexdigest()
            if (meta['arm'], meta['order'], meta['order_sha256'], meta['producer_source_sha256']) != (arm, order, order_hash, expected):
                raise ValueError(f'delta identity or pinned order mismatch: {name}')
            validate_delta(z['delta'], len(orders[order]), strict_zero=arm == 'Rzero')
        with src.open('rb') as reader, dst.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
        if digest(dst) != sha or digest(src) != sha:
            raise ValueError(f'delta changed during copy: {name}')
        files[name] = sha
    with output.open('x') as f:
        json.dump({'files': files, 'producer_source_sha256': expected,
                   'producer_manifest_sha256': digest(lock_path)}, f, indent=2, sort_keys=True)
        f.write('\n')
    print('88 locked seed-arm/order deltas staged; no score', flush=True)


if __name__ == '__main__':
    main()
