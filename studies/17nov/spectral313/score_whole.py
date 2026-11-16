"""Score only the complete verified §313 whole-model matrix."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ['S313_BASE']).resolve(strict=True)
CHAIN = Path(os.environ['S313_REFERENCE_CHAIN']).resolve(strict=True)
sys.path[:0] = [str(HERE), str(BASE / 'experiments'), str(CHAIN / 'studies/14nov/newmember')]
import protocol  # noqa: E402
from design import ARMS, ORDERS  # noqa: E402
from input_data import digest  # noqa: E402
from score_s_v66p import person_accuracy, summarize  # noqa: E402

PEOPLE = tuple(f'P{i:02d}' for i in range(1, 11))


def main():
    root = Path(os.environ['S313_COLLECT_OUT']).resolve(strict=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = {f'{a}_{o}.npz' for a in ARMS for o in ORDERS}
    if (set(manifest['files']['whole']) != expected or len(manifest['folds']) != 10 or
            len(manifest['gates']) != 10 or manifest['producer_source_sha256'] != digest(Path(os.environ['S313_MANIFEST'])) or
            manifest['collector_source_sha256'] != digest(Path(os.environ['S313_COLLECT_MANIFEST']))):
        raise ValueError('whole-model matrix/provenance incomplete')
    with np.load(BASE / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    whole = {}
    for arm in ARMS:
        for order in ORDERS:
            name = f'{arm}_{order}.npz'
            path = root / 'whole' / name
            if digest(path) != manifest['files']['whole'][name]:
                raise ValueError(f'{name}: whole output checksum mismatch')
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata'].item()))
                p = z['probability'].copy()
            if ((meta['arm'], meta['order'], meta['order_sha256'], meta['producer_source_sha256']) !=
                    (arm, order, hashlib.sha256(perms[order].tobytes()).hexdigest(), manifest['producer_source_sha256']) or
                    p.shape != (2655, 3) or not np.isfinite(p).all() or np.any(p < 0) or
                    not np.allclose(p.sum(1), 1, atol=1e-10)):
                raise ValueError(f'{name}: invalid order or probability')
            whole[arm, order] = p
    # Label access is deliberately after the complete-matrix and provenance checks.
    with np.load(BASE / 'data/proteus_windows.npz', allow_pickle=False) as z:
        y = z['y'].copy()
        subject, session, interface = (z[k].copy() for k in ('subject', 'session', 'interface'))
    with np.load(CHAIN / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        if not np.array_equal(z['y|all_loso'], y):
            raise ValueError('official and replay labels differ')
        cell, person = z['cell|all_loso'], z['person|all_loso']
    if any(set(y[cell == c]) != {0, 1, 2} for c in np.unique(cell)):
        raise ValueError('official cell lacks a class')
    d = {'subject': subject, 'session': session, 'interface': interface,
         'stream': np.char.add(np.char.add(subject, '|'), session)}
    official, by_person = {}, {}
    for arm in ARMS:
        for order in ORDERS:
            p = whole[arm, order]
            official[arm, order] = float(100 * protocol.score(y, p.argmax(1), d, np.ones(len(y), dtype=bool))['balanced_accuracy_cell_mean'])
            by_person[arm, order] = person_accuracy(p, y, cell, person, PEOPLE)
    pairs = [(a, 'BASE') for a in ARMS if a != 'BASE'] + [('IMAG', 'REAL'), ('BOTH', 'REAL')]
    results = {'official_equal_cell_by_order': {a: {o: official[a, o] for o in ORDERS} for a in ARMS},
               'official_equal_cell_scheme_equal': {a: float(np.mean([np.mean([official[a, o] for o in ORDERS if o.startswith(s)])
                   for s in ('within', 'full', 'runs')])) for a in ARMS},
               'absolute_people_order': {a: summarize(np.stack([by_person[a, o] for o in ORDERS]), ORDERS, PEOPLE) for a in ARMS},
               'contrasts_people_order': {f'{a}_minus_{b}': summarize(np.stack([by_person[a, o]-by_person[b, o] for o in ORDERS]), ORDERS, PEOPLE)
                                         for a,b in pairs},
               'manifest_sha256': digest(root / 'manifest.json'), 'source_sha256': manifest['producer_source_sha256'],
               'scope': 'whole new-person models; SPECTRAL is an independent static representation; no returning or package claim'}
    target = root / 'whole_results.json'
    with target.open('x') as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'complete guarded whole-model matrix scored: {target}', flush=True)


if __name__ == '__main__':
    main()
