"""Independent single-fold artifact guard; does not load any label array."""

import json
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design import ARMS, ORDERS, PARTS, VIEWS  # noqa: E402
from input_data import digest, load_spectral  # noqa: E402


def main(person):
    if person not in range(10):
        raise ValueError('invalid outer person')
    name = f'P{person+1:02d}'
    out = Path(os.environ['S313_OUT']).resolve(strict=True)
    path = out / 'folds' / f'{name}.npz'
    gate = out / 'gates' / f'{name}.json'
    if gate.exists():
        raise FileExistsError(gate)
    with np.load(Path(os.environ['S313_BASE']) / 'data/proteus_windows.npz', allow_pickle=False) as z:
        subject = z['subject'].copy()
    rows = np.flatnonzero(subject == f'Dreyer2026Proteus/{name}')
    _, spectral_sha = load_spectral()
    weights = {}
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata'].item()))
        if (metadata['person'] != name or metadata['device'] != 'CPU 0' or
                not np.array_equal(z['rows'], rows) or metadata['source_sha256'] != digest(Path(os.environ['S313_MANIFEST'])) or
                metadata['source_rows_sha256'] != hashlib.sha256(np.flatnonzero(subject != f'Dreyer2026Proteus/{name}').tobytes()).hexdigest() or
                metadata['solver_sha256'] != digest(Path(os.environ['S313_SOLVER_LIB'])) or
                metadata['spectral_manifest_sha256'] != spectral_sha or metadata['held_accuracy_computed'] is not False or
                not np.isfinite([metadata['prefix_max_abs'], metadata['reset_max_abs']]).all() or
                metadata['prefix_max_abs'] > 1e-10 or metadata['reset_max_abs'] > 1e-10 or
                set(metadata['measurements']) != set(ARMS)):
            raise ValueError('fold metadata, held identity or causal smoke mismatch')
        expected_cache = {f'P{i:02d}.npz': digest(Path(os.environ['S313_BASE']) / f'runs/s301_native_inputs/P{i:02d}.npz')
                          for i in range(1, 11)}
        if metadata['strong_cache_sha256'] != expected_cache:
            raise ValueError('strong input cache hash mismatch')
        for arm in ARMS:
            m = metadata['measurements'][arm]
            model_path = out / m['model_path']
            dim = sum(VIEWS[v] for v in PARTS[arm])
            if (m['features'] != dim or digest(model_path) != m['model_sha256'] or
                    not np.isfinite([m['loss'], m['independent_loss'], m['max_gradient'], m['independent_max_gradient']]).all() or
                    abs(m['loss']-m['independent_loss']) > 1e-9 or
                    max(m['max_gradient'], m['independent_max_gradient']) > 1e-5 or m['objective_evaluations'] <= 0):
                raise ValueError(f'{arm}: source model or objective guard failed')
            with np.load(model_path, allow_pickle=False) as model:
                theta = model['theta']
                mean, scale = model['mean'], model['scale']
                if (theta.shape != (3, dim+1) or mean.shape != (dim,) or scale.shape != (dim,) or
                        not np.isfinite(theta).all() or not np.isfinite(mean).all() or
                        not np.isfinite(scale).all() or np.any(scale <= 0)):
                    raise ValueError(f'{arm}: invalid model parameters')
                offset, norms = 0, {}
                for view in PARTS[arm]:
                    norms[view] = float(np.linalg.norm(theta[:, offset:offset+VIEWS[view]]))
                    offset += VIEWS[view]
                if any(not np.isfinite(value) or value <= 0 for value in norms.values()):
                    raise ValueError(f'{arm}: an expected representation view is inactive')
                weights[arm] = norms
            spectral_reference = None
            for order in ORDERS:
                p = z[f'prob|{arm}|{order}']
                if p.shape != (len(rows), 3) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-10):
                    raise ValueError(f'{arm}/{order}: invalid probability')
                if arm == 'SPECTRAL':
                    if spectral_reference is not None and not np.allclose(p, spectral_reference, atol=1e-12, rtol=0):
                        raise ValueError('spectral-only model changed with delivery order')
                    spectral_reference = p
                else:
                    delta = z[f'delta|{arm}|{order}']
                    if delta.shape != p.shape or not np.isfinite(delta).all() or not np.allclose(delta.sum(1), 0, atol=1e-10):
                        raise ValueError(f'{arm}/{order}: invalid posterior delta')
    gate.parent.mkdir(parents=True, exist_ok=True)
    with gate.open('x') as f:
        json.dump({'person': name, 'fold_sha256': digest(path), 'source_sha256': metadata['source_sha256'],
                   'view_weight_norms': weights, 'no_labels_read': True}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'{name}: independent source, active-view and eight-order artifact guard passed; no labels read', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: check.py OUTER_PERSON_INDEX')
    main(int(sys.argv[1]))
