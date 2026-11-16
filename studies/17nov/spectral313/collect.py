"""Assemble only the complete, source-locked ten-person model matrix."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from design import ARMS, ORDERS  # noqa: E402
from input_data import digest  # noqa: E402


def validate_fold_names(names):
    if set(names) != {f'P{i:02d}.npz' for i in range(1, 11)}:
        raise ValueError('incomplete or extra fold files')


def main():
    import check
    base = Path(os.environ['S313_BASE']).resolve(strict=True)
    producer = Path(os.environ['S313_OUT']).resolve(strict=True)
    output = Path(os.environ['S313_COLLECT_OUT'])
    if output.exists():
        raise FileExistsError(output)
    original_check = Path(os.environ['S313_MANIFEST']).parent / 'studies/17nov/spectral313/check.py'
    if digest(HERE / 'check.py') != digest(original_check):
        raise ValueError('collector guard differs from P01-smoked source')
    validate_fold_names({p.name for p in (producer / 'folds').glob('*.npz')})
    with np.load(base / 'data/proteus_windows.npz', allow_pickle=False) as z:
        subject = z['subject'].copy()
    n = len(subject)
    with np.load(base / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        permutations = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    if any(not np.array_equal(np.sort(p), np.arange(n)) for p in permutations.values()):
        raise ValueError('invalid pinned permutation')
    source_sha = digest(Path(os.environ['S313_MANIFEST']))
    whole = {(a, o): np.full((n, 3), np.nan) for a in ARMS for o in ORDERS}
    deltas = {(a, o): np.full((n, 3), np.nan) for a in ('Rzero', *ARMS[:-1]) for o in ORDERS}
    folds, gates = {}, {}
    for person in range(10):
        name = f'P{person+1:02d}'
        path, gate = producer / 'folds' / f'{name}.npz', producer / 'gates' / f'{name}.json'
        if not gate.exists():
            check.main(person)
        evidence = json.loads(gate.read_text())
        if (evidence['person'] != name or evidence['fold_sha256'] != digest(path) or
                evidence['source_sha256'] != source_sha or evidence['no_labels_read'] is not True):
            raise ValueError(f'{name}: stale or missing independent gate')
        with np.load(path, allow_pickle=False) as z:
            rows = z['rows']
            meta = json.loads(str(z['metadata'].item()))
            if not np.array_equal(rows, np.flatnonzero(subject == f'Dreyer2026Proteus/{name}')):
                raise ValueError('held rows changed')
            for arm in ARMS:
                model = meta['measurements'][arm]
                if digest(producer / model['model_path']) != model['model_sha256']:
                    raise ValueError(f'{name}/{arm}: model changed after gate')
                for order in ORDERS:
                    whole[arm, order][rows] = z[f'prob|{arm}|{order}']
                    if arm != 'SPECTRAL':
                        deltas[arm, order][rows] = z[f'delta|{arm}|{order}']
            for order in ORDERS:
                deltas['Rzero', order][rows] = 0
        folds[name] = digest(path)
        gates[name] = digest(gate)
    if any(not np.isfinite(a).all() for a in (*whole.values(), *deltas.values())):
        raise ValueError('incomplete whole or component matrix')
    if any(not np.any(deltas[arm, order] != 0) for arm in ARMS[:-1] for order in ORDERS):
        raise ValueError('inactive full-system replacement')
    (output / 'whole').mkdir(parents=True)
    (output / 'deltas').mkdir()
    files = {'whole': {}, 'deltas': {}}
    for category, data in (('whole', whole), ('deltas', deltas)):
        for (arm, order), array in data.items():
            target = output / category / f'{arm}_{order}.npz'
            metadata = {'arm': arm, 'order': order, 'order_sha256': hashlib.sha256(permutations[order].tobytes()).hexdigest(),
                        'producer_source_sha256': source_sha}
            with target.open('xb') as f:
                np.savez_compressed(f, **({'probability': array} if category == 'whole' else {'delta': array}),
                                    metadata=json.dumps(metadata, sort_keys=True))
            files[category][target.name] = digest(target)
    with (output / 'manifest.json').open('x') as f:
        json.dump({'files': files, 'folds': folds, 'gates': gates, 'producer_source_sha256': source_sha,
                   'collector_source_sha256': digest(Path(os.environ['S313_COLLECT_MANIFEST']))}, f, indent=2, sort_keys=True)
        f.write('\n')
    print('ten independently guarded folds assembled; no labels read', flush=True)


if __name__ == '__main__':
    main()
