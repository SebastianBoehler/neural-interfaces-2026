"""Stage a complete §313 component matrix into a private full-chain workspace."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'studies/16nov/followups/native_core')]
from chain_313 import ARMS, ORDERS  # noqa: E402
from input_data import digest  # noqa: E402
from member_chain import validate_delta  # noqa: E402


def main():
    source = Path(os.environ['S313_COLLECT_OUT']).resolve(strict=True)
    producer = Path(os.environ['S313_OUT']).resolve(strict=True)
    expected = os.environ['S313_SOURCE_SHA']
    if digest(Path(os.environ['S313_MANIFEST'])) != expected:
        raise ValueError('spectral producer source lock mismatch')
    manifest = source / 'manifest.json'
    lock = json.loads(manifest.read_text())
    if (lock['producer_source_sha256'] != expected or len(lock['folds']) != 10 or len(lock['gates']) != 10 or
            lock['collector_source_sha256'] != digest(Path(os.environ['S313_COLLECT_MANIFEST'])) or
            set(lock['files']['deltas']) != {f'{a}_{o}.npz' for a in ARMS for o in ORDERS}):
        raise ValueError('incomplete spectral component matrix')
    for name, sha in lock['folds'].items():
        if digest(producer / 'folds' / f'{name}.npz') != sha:
            raise ValueError(f'changed source fold: {name}')
        if digest(producer / 'gates' / f'{name}.json') != lock['gates'][name]:
            raise ValueError(f'changed independent fold gate: {name}')
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    target = ROOT / 'runs/s301_native/deltas'
    output = target.parent / 'manifest.json'
    if target.exists() or output.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    files = {}
    for name, sha in sorted(lock['files']['deltas'].items()):
        arm, order = name[:-4].split('_', 1)
        src, dst = source / 'deltas' / name, target / name
        if digest(src) != sha:
            raise ValueError(f'changed source delta: {name}')
        with np.load(src, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            order_sha = hashlib.sha256(perms[order].tobytes()).hexdigest()
            if (meta['arm'], meta['order'], meta['order_sha256'], meta['producer_source_sha256']) != (arm, order, order_sha, expected):
                raise ValueError(f'order or source mismatch: {name}')
            validate_delta(z['delta'], len(perms[order]), strict_zero=arm == 'Rzero')
        with src.open('rb') as reader, dst.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
        if digest(src) != sha or digest(dst) != sha:
            raise ValueError(f'copied delta checksum differs: {name}')
        files[name] = sha
    with output.open('x') as f:
        json.dump({'files': files, 'producer_source_sha256': expected,
                   'producer_manifest_sha256': digest(manifest)}, f, indent=2, sort_keys=True)
        f.write('\n')
    print('48 locked component/order deltas staged; no labels read', flush=True)


if __name__ == '__main__':
    main()
