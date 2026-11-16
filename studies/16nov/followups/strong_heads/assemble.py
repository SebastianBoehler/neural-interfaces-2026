"""Assemble the complete 20-fold/seed matrix without accessing outer labels."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ['S308_BASE']).resolve(strict=True)
OUT = Path(os.environ['S308_OUT']).resolve(strict=True)
sys.path[:0] = [str(HERE), str(BASE / 'experiments')]
import protocol  # noqa: E402
from definitions import ARMS, ORDERS, SEED_ARMS  # noqa: E402
from produce import digest  # noqa: E402
from split import source_partition  # noqa: E402


def main():
    output = OUT / 'deltas'
    manifest = OUT / 'manifest.json'
    if output.exists() or manifest.exists():
        raise FileExistsError('assembly output already exists')
    d = protocol.load()
    people = np.array([int(s.rsplit('P', 1)[1]) - 1 for s in d['subject']])
    n = len(people)
    with np.load(BASE / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    if any(not np.array_equal(np.sort(p), np.arange(n)) for p in perms.values()):
        raise ValueError('invalid pinned permutation')
    source_hash = digest(Path(os.environ['S308_MANIFEST']))
    input_hash = {f'P{p + 1:02d}.npz': digest(Path(os.environ['S308_CACHE']) / f'P{p + 1:02d}.npz') for p in range(10)}
    reference = Path(os.environ['S308_REFERENCES'])
    lib_hash = digest(Path(os.environ['S308_LIB']))
    full_arms = SEED_ARMS
    arrays = {(arm, order): np.full((n, 3), np.nan) for arm in full_arms for order in ORDERS}
    shards = {}
    for person in range(10):
        for seed in (0, 1):
            name = f'P{person + 1:02d}_s{seed}.npz'
            path = OUT / 'shards' / name
            before = digest(path)
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'].item()))
                _, inner, outer = source_partition(people, person, seed)
                rows = z['rows']
                if (meta['outer_person'] != person or meta['inner_person'] != int(np.unique(people[inner])[0]) or
                        meta['seed'] != seed or not np.array_equal(rows, np.flatnonzero(outer)) or
                        meta['source_sha256'] != source_hash or meta['cache_sha256'] != input_hash or
                        meta['head_library_sha256'] != lib_hash or meta['held_accuracy_computed'] is not False or
                        meta['reference_sha256'] != digest(reference / f'P{person + 1:02d}.npz') or
                        set(meta['models']) != set(ARMS) or set(meta['selection']) != set(ARMS)):
                    raise ValueError(f'{name}: fold or source provenance mismatch')
                for arm in ARMS:
                    model = meta['models'][arm]
                    path_model = OUT / model['path']
                    if digest(path_model) != model['sha256'] or not np.isfinite(model['seconds']):
                        raise ValueError(f'{name}/{arm}: checkpoint mismatch')
                    for order in ORDERS:
                        value = z[f'{arm}|{order}']
                        if (value.shape != (len(rows), 3) or not np.isfinite(value).all() or
                                not np.allclose(value.sum(1), 0, atol=1e-10, rtol=0)):
                            raise ValueError(f'{name}/{arm}/{order}: invalid delta')
                        arrays[f'{arm}{seed}', order][rows] = value
                for order in ORDERS:
                    arrays['Rzero', order][rows] = 0
            if digest(path) != before:
                raise ValueError('shard changed during assembly')
            shards[name] = before
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('incomplete held-person matrix')
    if any(not np.any(arrays[a, o] != 0) for a in full_arms[1:] for o in ORDERS):
        raise ValueError('an entire active head/order is a no-op')
    output.mkdir(parents=True)
    files = {}
    for (arm, order), delta in arrays.items():
        path = output / f'{arm}_{order}.npz'
        metadata = {'arm': arm, 'order': order, 'order_sha256': hashlib.sha256(perms[order].tobytes()).hexdigest(),
                    'producer_source_sha256': source_hash}
        with path.open('xb') as f:
            np.savez_compressed(f, delta=delta, metadata=json.dumps(metadata, sort_keys=True))
        files[path.name] = digest(path)
    with manifest.open('x') as f:
        json.dump({'files': files, 'shards': shards, 'producer_source_sha256': source_hash,
                   'head_library_sha256': lib_hash, 'reference_manifest_sha256': digest(reference / 'manifest.json')},
                  f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'{len(files)} ordered head deltas assembled from 20 verified shards; no score', flush=True)


if __name__ == '__main__':
    main()
