"""Replace only the terminal stage with its C++ implementation."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from variants import STAGES


def digest(name):
    with Path(name).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def sdk():
    torch.set_num_threads(1)
    ct.CDLL('/stock/libni26_base.so', mode=ct.RTLD_GLOBAL)
    lib = ct.CDLL('/work/witness.so')
    lib.ni26_cp_create.argtypes = [ct.c_char_p]
    lib.ni26_cp_create.restype = ct.c_void_p
    lib.ni26_cp_step.argtypes = [ct.c_void_p]*5
    lib.ni26_cp_step.restype = ct.c_int
    lib.ni26_cp_stats.argtypes = [ct.c_void_p]*2
    lib.ni26_cp_stats.restype = None
    lib.ni26_cp_destroy.argtypes = [ct.c_void_p]
    lib.ni26_cp_destroy.restype = None
    return lib


def evaluate(lib, raw, index, bounds, outer, counted, arm):
    target = Path('/stock') if arm == 'stock' else Path('/work')/arm
    scores = np.empty_like(counted)
    fits = 0
    start = perf_counter()
    for start_row, stop_row in zip(bounds[:-1], bounds[1:], strict=True):
        model = lib.ni26_cp_create(str(target/'native.pt').encode())
        if not model:
            raise RuntimeError('cannot instantiate native personalization')
        try:
            for i in index[start_row:stop_row]:
                if lib.ni26_cp_step(model, raw[i].ctypes.data, outer[i].ctypes.data,
                                   counted[i].ctypes.data, scores[i].ctypes.data):
                    raise RuntimeError(f'native personalization step failed at row {i}')
            stats = np.zeros(4, dtype=np.int64)
            lib.ni26_cp_stats(model, stats.ctypes.data)
        finally:
            lib.ni26_cp_destroy(model)
        n = int(stop_row-start_row)
        wanted = [n, len(range(STAGES[arm], n, 20)), max(0, n-90), 0]
        if stats.tolist() != wanted:
            raise RuntimeError(f'{arm} work differs: {stats} vs {wanted}')
        fits += wanted[1]
    return scores, {'refits': fits, 'seconds': perf_counter()-start}


def main():
    lib = sdk()
    with np.load('/data.npz', allow_pickle=False) as f:
        raw = np.ascontiguousarray(f['X'], dtype=np.float32)
    if raw.shape != (2655, 41, 480):
        raise ValueError('public EEG input differs')
    output, proof = {}, {}
    with np.load('inputs.npz', allow_pickle=False) as f:
        for oi, order in enumerate(f['orders']):
            index, bounds = (f[f'{order}|{key}'] for key in ('index', 'bounds'))
            vectors = [np.ascontiguousarray(f[f'{order}|{key}'], dtype=np.float64)
                       for key in ('raw', 'filtered', 'expected')]
            outer, counted, expected = vectors
            if (not np.array_equal(np.sort(index), np.arange(2655)) or len(bounds) != 15
                    or not all(a.shape == (2655, 3) for a in vectors)):
                raise ValueError('held-source fixture shape differs')
            methods = list(STAGES)
            methods = methods[oi % 3:] + methods[:oi % 3]
            order_output, trace = {}, {}
            for arm in methods:
                order_output[arm], trace[arm] = evaluate(lib, raw, index, bounds, outer, counted, arm)
            stock, skip, phase = (order_output[a] for a in STAGES)
            same = np.array_equal(stock.argmax(1), expected.argmax(1))
            error = float(np.max(np.abs(stock-expected)))
            if not same or error > 1e-7 or not np.array_equal(stock, skip):
                raise ValueError(f'{order}: stock/skip qualification failed ({same=}, {error=})')
            if not np.isfinite(phase).all() or np.array_equal(stock, phase):
                raise ValueError(f'{order}: phase90 made no score change')
            proof[str(order)] = dict(trace=trace, stock_max_error=error,
                                     phase90_changed=int(np.sum(stock.argmax(1) != phase.argmax(1))))
            output.update({f'{order}|{a}': v for a, v in order_output.items()})
            print(order, proof[str(order)], flush=True)
    if len(proof) != 8:
        raise ValueError('expected eight held-DEV orders')
    np.savez_compressed('outputs.npz', **output)
    Path('before_labels.json').write_text(json.dumps(dict(
        input=digest('inputs.npz'), predictions=digest('outputs.npz'),
        orders=proof, held_labels_opened=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
