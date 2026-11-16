"""Stage complete producer matrices as explicit two-seed ensemble members."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from config import GROUPS, ORDERS, sha

PRODUCERS = {
 '312': Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2'),
 '314': Path('/mnt/lustre/work/gehler/gfh527/ni26_s314_time_patterns_20261117/production'),
}
SOURCES = {'312': '8275d60d30d618a1a90d184f504522682b252221e956356e5041bed2b3ac6559',
           '314': 'baa34aa59fc34c0722dcf0c65a02b6d1fc1b70e93e32190b5ac7712dde49af0a'}


def valid(p, n):
    if p.shape != (n, 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('invalid canonical probabilities')


def load312(source, subjects, inputs):
    result = {}
    summary = source / 'whole_results.json'
    if not summary.is_file():
        raise ValueError('successful whole312collector required')
    report = json.loads(summary.read_text()); inputs[str(summary)] = sha(summary)
    expected = {f'{a}_{arm}_s{s}_P{p:02d}.npz' for a in ('fb', 'sh') for arm in ('S', 'F', 'T', 'C') for s in (0, 1) for p in range(1, 11)}
    if set(report['hashes']) != expected:
        raise ValueError('whole312collector did not cover complete160foldmatrix')
    for name, digest in report['hashes'].items():
        path = source / 'output' / name
        if sha(path) != digest:
            raise ValueError('312complete matrix changed after collector')
        inputs[str(path)] = digest
    for arch, arm, family in (('fb', 'T', 'fbt312'), ('fb', 'C', 'fbc312'), ('fb', 'S', 'fbs312'),
                              ('sh', 'T', 'sht312'), ('sh', 'C', 'shc312'), ('sh', 'S', 'shs312')):
        for seed in (0, 1):
            for order in ORDERS:
                result[family, seed, order] = np.full((len(subjects), 3), np.nan)
            for person in range(1, 11):
                held = f'P{person:02d}'; path = source / 'output' / f'{arch}_{arm}_s{seed}_{held}.npz'
                with np.load(path, allow_pickle=False) as z:
                    m = json.loads(str(z['metadata'].item()))
                    rows = np.flatnonzero(np.char.endswith(subjects, '/' + held))
                    target = subjects[rows[0]]
                    if (m['architecture'], m['arm'], m['seed'], m['held']) != (arch, arm, seed, held):
                        raise ValueError('312arm identity differs')
                    if not np.array_equal(z['index'], rows) or target in m['fit_people'] or target in m['inner_fit_people']:
                        raise ValueError('312source exclusion/row guard failed')
                    if len(m['fit_people']) != 9 or len(m['inner_fit_people']) != 8 or m['source_manifest'] != SOURCES['312']:
                        raise ValueError('312source provenance differs')
                    checkpoint = path.with_suffix('.pt')
                    if sha(checkpoint) != m['checkpoint_sha256']:
                        raise ValueError('312checkpoint changed')
                    inputs[str(checkpoint)] = m['checkpoint_sha256']
                    for raw, digest in m['raw_hashes'].items():
                        if raw not in inputs:
                            if sha(raw) != digest:
                                raise ValueError('312raw input changed')
                            inputs[raw] = digest
                    for order in ORDERS:
                        p = z[f'proba_{order}']; valid(p, len(rows))
                        result[family, seed, order][rows] = p
    return result


def load314(source, n, inputs):
    summary = source / 'evidence/next_wave_16nov/s314_results.json'
    if not summary.is_file():
        raise ValueError('successful owning whole314collector required')
    inputs[str(summary)] = sha(summary)
    folder = source / 'runs/s314/streams'
    validation = folder / 'validation.json'
    m = json.loads(validation.read_text())
    if m['source_sha256'] != SOURCES['314'] or len(m['folds']) != 20 or m['held_accuracy_computed']:
        raise ValueError('314assembler provenance differs')
    inputs[str(validation)] = sha(validation)
    for name, digest in m['folds'].items():
        path = source / 'runs/s314/folds' / name
        if sha(path) != digest:
            raise ValueError('314source fold changed')
        inputs[str(path)] = digest
    result = {}
    for family in GROUPS['314']['families']:
        for seed in (0, 1):
            for order in ORDERS:
                name = f'ttt18{order}_{family}_base-all_loso-s{seed}.npz'; path = folder / name
                if sha(path) != m['streams'][name]:
                    raise ValueError('314stream hash differs')
                inputs[str(path)] = sha(path)
                with np.load(path, allow_pickle=False) as z:
                    info = json.loads(str(z['metadata'].item()))
                    if not np.array_equal(z['index'], np.arange(n)) or (info['seed'], info['order']) != (seed, order):
                        raise ValueError('314canonical rows/order differ')
                    if info['fold_sha256'] != m['folds'] or info['source_sha256'] != SOURCES['314']:
                        raise ValueError('314source/fold provenance differs')
                    valid(z['proba'], n); result[family, seed, order] = z['proba'].copy()
    if len(m['streams']) != 32:
        raise ValueError('incomplete314stream matrix')
    return result


def main(group):
    root = Path(os.environ['NI26_WS']).resolve(strict=True); source = PRODUCERS[group]
    manifest = source / ('source.sha256' if group == '312' else 's314-source.sha256')
    if sha(manifest) != SOURCES[group]:
        raise ValueError('pinned producer source manifest changed')
    inputs = {str(manifest): SOURCES[group]}
    with np.load(root / 'data/proteus_windows.npz') as z:
        subjects = z['subject'].astype(str)
    n = len(subjects)
    result = load312(source, subjects, inputs) if group == '312' else load314(source, n, inputs)
    directory = root / 'runs/representation_streams'; directory.mkdir(exist_ok=False)
    target = root / 'runs/representation_inputs'; target.mkdir(exist_ok=False)
    records = {}
    for family in GROUPS[group]['families']:
        for order in ORDERS:
            p = np.mean([result[family, seed, order] for seed in (0, 1)], axis=0); valid(p, n)
            metadata = dict(family=family, order=order, seeds=[0, 1], operation='arithmetic_probability_mean',
                            use='same fixed ensemble in all six dependent incumbent CP bases', producer_source=SOURCES[group])
            path = directory / f'ttt18{order}_{family}_base-all_loso-ensemble2.npz'
            with path.open('xb') as f:
                np.savez(f, index=np.arange(n), proba=p, metadata=json.dumps(metadata))
            records[path.name] = sha(path)
    output = target / 'streams.sha256'
    with output.open('x') as f:
        for name, digest in sorted(records.items()):
            f.write(f'{digest}  {name}\n')
    with (target / 'lock.json').open('x') as f:
        json.dump(dict(manifest_sha256=sha(output), producer_inputs_sha256=inputs, group=group), f, indent=2)
    print(f'complete{group}source matrix staged as{len(records)}explicit two-seed ensemble streams;no scores read')


if __name__ == '__main__':
    main(sys.argv[1])
