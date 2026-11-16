"""Assemble only the complete, provenance-checked native-member matrix."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'studies/16nov/followups/geometry')]
import protocol  # noqa: E402
from config import ARCHIVES  # noqa: E402

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
ARMS = ('Rzero', 'Nfixed', 'Ntrain')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    out = ROOT / 'runs/s301_native/deltas'
    if out.exists():
        raise FileExistsError(out)
    d = protocol.load()
    n = len(d['subject'])
    folds = {held: np.flatnonzero(te) for held, _, te in protocol.folds(d, 'all_loso')}
    names = {f'P{i:02d}' for i in range(1, 11)}
    if set(folds) != names:
        raise ValueError('native fold matrix incomplete')
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    if any(not np.array_equal(np.sort(p), np.arange(n)) for p in perms.values()):
        raise ValueError('invalid pinned order')
    archive_hash = {p: sha(ROOT / 'runs' / p) for p in sorted(set(ARCHIVES.values()))}
    input_hash = {f'{p}.npz': sha(ROOT / f'runs/s301_native_inputs/{p}.npz') for p in sorted(names)}
    data_sha, perm_sha = sha(ROOT / 'data/proteus_windows.npz'), sha(ROOT / 'runs/shuffle_perms.npz')
    source_sha = sha(ROOT / 'producer.sha256')
    reference = Path(os.environ['S301_ALL_REFERENCES']).resolve(strict=True)
    parity_sha = sha(Path(os.environ['S301_PARITY_REPORT']))
    arrays = {(a, o): np.full((n, 3), np.nan) for a in ARMS for o in ORDERS}
    hashes = {}
    for person in sorted(names):
        path = ROOT / f'runs/s301_native_shards/{person}.npz'
        before = sha(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            rows = z['rows']
            if (meta['held'] != person or not np.array_equal(rows, folds[person]) or
                    meta['source_sha256'] != source_sha or meta['input_sha256'] != input_hash or
                    meta['reference_sha256'] != sha(reference / f'{person}.npz') or
                    meta['parity_sha256'] != parity_sha or meta['data_sha256'] != data_sha or
                    meta['permutation_sha256'] != perm_sha or meta['archives_sha256'] != archive_hash or
                    meta['held_accuracy_computed'] is not False or
                    not np.isfinite([meta['native_fit_loss'], meta['reference_loss'], meta['max_gradient']]).all() or
                    abs(meta['native_fit_loss'] - meta['reference_loss']) > 1e-5 or meta['max_gradient'] > 1e-5):
                raise ValueError(f'{person}: source, fold or training guard failed')
            for order in ORDERS:
                for arm in ('Nfixed', 'Ntrain'):
                    value = z[f'{arm}|{order}']
                    if value.shape != (len(rows), 3) or not np.isfinite(value).all() or not np.allclose(value.sum(1), 0, atol=1e-10, rtol=0):
                        raise ValueError(f'{person}/{arm}/{order}: invalid native delta')
                    arrays[arm, order][rows] = value
                arrays['Rzero', order][rows] = 0
        if sha(path) != before:
            raise ValueError('native shard changed during collection')
        hashes[path.name] = before
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError('incomplete native output rows')
    out.mkdir(parents=True)
    files = {}
    for (arm, order), delta in arrays.items():
        path = out / f'{arm}_{order}.npz'
        metadata = {'arm': arm, 'order': order, 'order_sha256': hashlib.sha256(perms[order].tobytes()).hexdigest(),
                    'producer_source_sha256': source_sha}
        with path.open('xb') as f:
            np.savez_compressed(f, delta=delta, metadata=json.dumps(metadata, sort_keys=True))
        files[path.name] = sha(path)
    manifest = out.parent / 'manifest.json'
    with manifest.open('x') as f:
        json.dump({'files': files, 'shards': hashes, 'producer_source_sha256': source_sha,
                   'parity_sha256': parity_sha, 'reference_manifest_sha256': sha(reference / 'manifest.json')}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'assembled 24 locked native member deltas: {sha(manifest)}; no score', flush=True)


if __name__ == '__main__':
    main()
