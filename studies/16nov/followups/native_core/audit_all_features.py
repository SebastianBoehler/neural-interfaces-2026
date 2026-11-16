"""No-score all-person native feature and per-fold scaler parity audit."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
sys.path.insert(0, str(ROOT / 'experiments'))
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    target = ROOT / 'runs/s305_feature_audit/report.json'
    if target.exists():
        raise FileExistsError(target)
    d = protocol.load(); n = len(d['subject'])
    ref = T.Feats(d, T.parse('car_ov17_fs2_t0.1_le_k2')).F
    native = np.empty_like(ref)
    seen = np.zeros(n, bool); person = {}; input_hashes = {}
    original = Path(os.environ['S301_INPUT_PRODUCER']).resolve(strict=True)
    original_sha = sha(original / 'producer.sha256')
    for i in range(1, 11):
        name = f'P{i:02d}'
        path = original / f'runs/s301_native_inputs/{name}.npz'
        input_hashes[path.name] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item())); rows = z['rows']; values = z['features']
            if (meta['person'] != name or meta['source_sha256'] != original_sha or meta['held_accuracy_computed'] is not False or
                    meta['rows_sha256'] != hashlib.sha256(rows.tobytes()).hexdigest() or values.shape != (len(rows), 44239) or
                    np.any(rows < 0) or np.any(rows >= n) or seen[rows].any()):
                raise ValueError(f'{name}: feature identity/provenance mismatch')
            diff = values - ref[rows]
            maximum = float(np.max(np.abs(diff))); relative = float(np.linalg.norm(diff) / np.linalg.norm(ref[rows]))
            if maximum > 1e-4 or relative > 1e-6:
                raise ValueError(f'{name}: native training feature mismatch {maximum=} {relative=}')
            person[name] = {'max_abs': maximum, 'relative_l2': relative}
            native[rows] = values; seen[rows] = True
    if not seen.all():
        raise ValueError('native training feature rows are incomplete')
    folds = {}
    for held, tr, te in protocol.folds(d, 'all_loso'):
        if np.any(tr & te):
            raise ValueError('training/held masks overlap')
        r, x = StandardScaler().fit(ref[tr]), StandardScaler().fit(native[tr])
        mean = float(np.max(np.abs(r.mean_ - x.mean_))); scale = float(np.max(np.abs(r.scale_ - x.scale_)))
        if mean > 1e-4 or scale > 1e-4:
            raise ValueError(f'{held}: native/reference scaler mismatch {mean=} {scale=}')
        folds[held] = {'mean_max_abs': mean, 'scale_max_abs': scale}
    if set(folds) != {f'P{i:02d}' for i in range(1, 11)}:
        raise ValueError('ten scaler folds required')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as f:
        json.dump({'people': person, 'fold_scalers': folds, 'input_sha256': input_hashes,
                   'producer_source_sha256': original_sha, 'audit_source_sha256': sha(ROOT / 'feature_audit.sha256'),
                   'held_accuracy_computed': False}, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'all ten people and fold scalers passed: {target}; no score', flush=True)


if __name__ == '__main__':
    main()
