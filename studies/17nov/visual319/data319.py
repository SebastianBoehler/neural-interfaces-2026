"""Source-only display statistics and explicit exclusion of P01/P02 from fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np

DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
DATA = Path('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/data/proteus_windows.npz')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load():
    with np.load(DATA, allow_pickle=False) as z:
        x, subject, names = z['X'], z['subject'].astype(str), z['ch_names'].astype(str)
    if x.shape != (2655, 41, 480) or not np.isfinite(x).all():
        raise ValueError('official source windows differ')
    fit = np.flatnonzero(np.isin(np.array([s.split('/')[-1] for s in subject]), [f'P{i:02d}' for i in range(3, 11)]))
    val = np.flatnonzero(np.char.endswith(subject, '/P02'))
    if len(set(subject[fit])) != 8 or not len(val) or np.any(np.char.endswith(subject[np.r_[fit, val]], '/P01')):
        raise ValueError('source partition/exclusion failed')
    return x, subject, names, fit, val


def fit_display(root, x, subject, names, fit):
    path = root / 'renderer.npz'
    if path.exists():
        raise FileExistsError(path)
    a = x[fit]
    scale = float(np.quantile(np.abs(a), .995))
    matrix = a.transpose(1, 0, 2).reshape(41, -1).astype(np.float64)
    moment = matrix @ matrix.T / matrix.shape[1]
    w, vectors = np.linalg.eigh(moment)
    projection = vectors[:, np.argsort(w)[::-1][:16]].T.copy()
    for row in projection:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    if not np.allclose(projection @ projection.T, np.eye(16), atol=1e-10):
        raise ValueError('projection is not orthonormal')
    meta = dict(fit_people=sorted(set(subject[fit])), excluded=['P01', 'P02'],
        source_rows_sha256=digest(fit), source_values_sha256=digest(a), channels=names.tolist(),
        scale=scale, scale_recipe='source absolute99.5percentile;no per-window normalization',
        projection_recipe='uncentered source second-moment top16;deterministic signs',
        projection_sha256=digest(projection), labels_opened=False)
    with path.open('xb') as out:
        np.savez(out, scale=scale, projection=projection, metadata=json.dumps(meta))
    path.chmod(0o444)
    return scale, projection, meta


def load_display(root, fit):
    with np.load(root / 'renderer.npz', allow_pickle=False) as z:
        scale, projection, meta = float(z['scale']), z['projection'].copy(), json.loads(str(z['metadata']))
    if meta['source_rows_sha256'] != digest(fit) or meta['projection_sha256'] != digest(projection) or meta['labels_opened']:
        raise ValueError('renderer/source provenance differs')
    return scale, projection, meta
