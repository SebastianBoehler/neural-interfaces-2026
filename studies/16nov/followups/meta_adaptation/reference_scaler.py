"""Load the pinned source-only scaler used by the independent §305 fit."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_scaler(root, held, train_rows, expected_sha, dimensions):
    root = Path(root).resolve(strict=True)
    path = root / f'{held}.npz'
    manifest = json.loads((root / 'manifest.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if (manifest.get('held_accuracy_computed') is not False or
            manifest.get('files', {}).get(path.name) != digest or digest != expected_sha):
        raise ValueError('§305 reference scaler provenance mismatch')
    with np.load(path, allow_pickle=False) as z:
        rows, mean, scale = (np.ascontiguousarray(z[k]) for k in ('fit_rows', 'mean', 'scale'))
    if (not np.array_equal(rows, train_rows) or mean.shape != (dimensions,) or
            scale.shape != (dimensions,) or not np.isfinite(mean).all() or
            not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('invalid scaler or source-only training rows')
    return mean, scale
