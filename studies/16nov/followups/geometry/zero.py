"""Create registered all-zero adapter controls without reading held labels."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain import ORDERS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(os.environ['NI26_WS']).resolve(strict=True)
    out = root / 'runs/s295_geometry/deltas'; out.mkdir(parents=True, exist_ok=True)
    manifest = out.parent / 'zero_manifest.json'
    if manifest.exists() or any((out / f'geo_zero_{o}.npz').exists() for o in ORDERS):
        raise FileExistsError('geometry zero output already exists')
    with np.load(root / 'data/proteus_windows.npz', allow_pickle=False) as z:
        n = len(z['subject'])
    files = {}
    with np.load(root / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        for order in ORDERS:
            perm = z[f'{order[:-1]}|{order[-1]}|all_loso']
            if not np.array_equal(np.sort(perm), np.arange(n)):
                raise ValueError(f'invalid order {order}')
            path = out / f'geo_zero_{order}.npz'
            meta = {'arm': 'geo_zero', 'order': order, 'order_sha256': hashlib.sha256(perm.tobytes()).hexdigest()}
            with path.open('xb') as f:
                np.savez_compressed(f, delta=np.zeros((n, 3)), metadata=json.dumps(meta, sort_keys=True))
            files[path.name] = sha(path)
    with manifest.open('x') as f:
        json.dump({'files': files}, f, sort_keys=True, indent=2); f.write('\n')
    print(f'zero manifest {manifest}: {sha(manifest)}')


if __name__ == '__main__':
    main()
