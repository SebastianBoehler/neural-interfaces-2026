"""Read only selected source label entries from the compressed dataset."""
import zipfile
import numpy as np


def read_selected(path, key, rows, total):
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or not len(rows):
        raise ValueError('invalid source-row indices')
    if np.any(rows < 0) or np.any(rows >= total) or np.any(np.diff(rows) <= 0):
        raise ValueError('source rows must be sorted, unique and in range')
    if key not in ('y', 'interface'):
        raise ValueError('unsupported training metadata')
    with zipfile.ZipFile(path) as archive, archive.open(f'{key}.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0): shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)): shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else: raise ValueError(f'unsupported NPY version {version}')
        allowed = dtype.kind in 'iu' and dtype.itemsize in (1, 2, 4, 8) if key == 'y' else dtype.kind in 'US'
        if shape != (total,) or fortran or not allowed:
            raise ValueError('unexpected source label array')
        start = stream.tell(); result = np.empty(len(rows), dtype=dtype)
        for i, row in enumerate(rows):
            stream.seek(start + int(row) * dtype.itemsize)
            raw = stream.read(dtype.itemsize)
            if len(raw) != dtype.itemsize: raise ValueError('truncated label array')
            result[i] = np.frombuffer(raw, dtype=dtype, count=1)[0]
    if key == 'y' and np.any((result < 0) | (result > 2)): raise ValueError('invalid source label')
    if key == 'interface' and set(result) != {'Graz', 'BrainHero'}: raise ValueError('invalid source interfaces')
    return result


def read_training_labels(path, rows, total):
    return read_selected(path, 'y', rows, total).astype(np.int64)
