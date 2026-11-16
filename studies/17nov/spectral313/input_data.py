"""Load source-only cached views and the held person's causal strong blocks."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_spectral():
    root = Path(os.environ['S313_FEATURES']).resolve(strict=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest['source_sha256'] != os.environ['S313_FEATURE_SOURCE_SHA'] or
            manifest['data_sha256'] != os.environ['S313_DATA_SHA'] or manifest['no_labels_read'] is not True):
        raise ValueError('spectral feature source or data lock mismatch')
    out = {}
    for name, dim in (('power', 861), ('real', 17220), ('imag', 17220)):
        path = root / f'{name}.npy'
        if digest(path) != manifest['files_sha256'][name]:
            raise ValueError(f'{name}: spectral cache checksum mismatch')
        out[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        if out[name].shape != (2655, dim) or out[name].dtype != np.float32:
            raise ValueError(f'{name}: spectral cache shape mismatch')
    return out, digest(root / 'manifest.json')


def load_strong(d, held):
    root = Path(os.environ['S313_BASE']).resolve(strict=True)
    cache = root / 'runs/s301_native_inputs'
    expected = digest(root / 'producer.sha256')
    parts, rows, priors, counts, hashes, selected = [], [], [], [], {}, None
    for p in range(10):
        path = cache / f'P{p+1:02d}.npz'
        hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            index = z['rows'].copy()
            person = f'Dreyer2026Proteus/P{p+1:02d}'
            if (meta['person'] != f'P{p+1:02d}' or meta['source_sha256'] != expected or
                    meta['rows_sha256'] != hashlib.sha256(index.tobytes()).hexdigest() or
                    meta['native_core_sha256'] != digest(Path(os.environ['NATIVE_CORE_LIB'])) or
                    meta['native_train_sha256'] != digest(Path(os.environ['NATIVE_TRAIN_LIB'])) or
                    meta['held_accuracy_computed'] is not False or
                    not np.array_equal(index, np.flatnonzero(d['subject'] == person))):
                raise ValueError(f'{person}: cached feature provenance mismatch')
            priors.append(z['prior'].copy())
            counts.append(len(index))
            if p == held:
                selected = {'rows': index, 'cov': z['cov'].copy(), 'slow': z['slow'].copy()}
            else:
                rows.append(index)
                parts.append(z['features'].copy())
    order = np.argsort(np.concatenate(rows))
    train_rows = np.concatenate(rows)[order]
    train = np.ascontiguousarray(np.concatenate(parts)[order])
    if selected is None or not np.array_equal(train_rows, np.flatnonzero(d['subject'] != f'Dreyer2026Proteus/P{held+1:02d}')):
        raise ValueError('source/held person split mismatch')
    return train, train_rows, selected, np.asarray(priors), np.asarray(counts), hashes
