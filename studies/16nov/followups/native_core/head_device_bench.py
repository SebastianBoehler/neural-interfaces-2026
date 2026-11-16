"""§309 resident-device objective guard and paired source-only head benchmark."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from scipy.special import logsumexp


PD = ctypes.POINTER(ctypes.c_double)
PI = ctypes.POINTER(ctypes.c_int64)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training():
    root = Path(os.environ['S309_PRODUCER']).resolve(strict=True)
    sys.path.insert(0, str(root / 'experiments'))
    os.environ['NI26_ROOT'] = str(root)
    import protocol
    d = protocol.load()
    fold = [(tr, te) for p, tr, te in protocol.folds(d, 'all_loso') if p == 'P01']
    if len(fold) != 1:
        raise ValueError('expected one P01 outer fold')
    tr, te = fold[0]
    parts, rows, hashes = [], [], {}
    source = sha(root / 'producer.sha256')
    for path in (root / 'producer.sha256', root / 'experiments/protocol.py', root / 'data/proteus_windows.npz'):
        hashes[str(path)] = sha(path)
    for i in range(2, 11):
        path = root / f'runs/s301_native_inputs/P{i:02d}.npz'
        hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            if (meta['person'] != f'P{i:02d}' or meta['source_sha256'] != source or
                    meta['held_accuracy_computed'] is not False):
                raise ValueError('source-person cache provenance differs')
            rows.append(z['rows'].copy())
            parts.append(z['features'].copy())
    index = np.concatenate(rows)
    position = np.argsort(index)
    if not np.array_equal(index[position], np.flatnonzero(tr)) or np.any(te[index]):
        raise ValueError('training rows include outer-held person or omit source rows')
    raw = np.ascontiguousarray(np.concatenate(parts)[position])
    y = np.ascontiguousarray(d['y'][index[position]], dtype=np.int64)
    ref_path = Path(os.environ['S309_REFERENCES']) / 'P01.npz'
    hashes[str(ref_path)] = sha(ref_path)
    with np.load(ref_path, allow_pickle=False) as ref:
        x = np.ascontiguousarray((raw - ref['mean']) / ref['scale'])
    if x.shape[1] != 44239 or not np.isfinite(x).all() or set(y) != {0, 1, 2}:
        raise ValueError('strong source-only training matrix invalid')
    return x, y, hashes


def objective(x, y, theta):
    w, b = theta[:, :-1], theta[:, -1]
    logits = x @ w.T + b
    normalizer = logsumexp(logits, axis=1)
    loss = np.mean(normalizer - logits[np.arange(len(y)), y])
    loss += np.sum(w * w) / (2 * .3 * len(y))
    p = np.exp(logits - normalizer[:, None])
    p[np.arange(len(y)), y] -= 1
    grad = np.column_stack((p.T @ x / len(y) + w / (.3 * len(y)), p.mean(0)))
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise ValueError('independent objective/gradient nonfinite')
    return float(loss), float(np.abs(grad).max())


class Head:
    def __init__(self):
        self.lib = ctypes.CDLL(os.environ['S309_LIBRARY'])
        signature = [PD, PI, ctypes.c_int64, ctypes.c_int64, ctypes.c_double]
        outputs = [PD, PD, PD, PI]
        self.lib.nc_head_fit_device.argtypes = signature + [ctypes.c_int] + outputs
        self.lib.nc_head_fit.argtypes = signature + outputs
        self.lib.nc_head_error.restype = ctypes.c_char_p

    def fit(self, x, y, device, legacy=False):
        theta = np.empty((3, x.shape[1] + 1), np.float64)
        loss, grad, calls = ctypes.c_double(), ctypes.c_double(), ctypes.c_int64()
        args = [x.ctypes.data_as(PD), y.ctypes.data_as(PI), len(y), x.shape[1], .3]
        args += [] if legacy else [device]
        args += [theta.ctypes.data_as(PD), ctypes.byref(loss), ctypes.byref(grad), ctypes.byref(calls)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        result = (self.lib.nc_head_fit if legacy else self.lib.nc_head_fit_device)(*args)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if result:
            raise RuntimeError(self.lib.nc_head_error().decode())
        check_loss, check_grad = objective(x, y, theta)
        if abs(check_loss - loss.value) > 1e-9 or check_grad > 1e-5:
            raise ValueError('independent loss/gradient guard failed')
        return theta, {'device': device, 'wall_s': elapsed, 'loss': check_loss,
                       'max_gradient': check_grad, 'objective_evaluations': calls.value,
                       'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'benchmark'])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('allocated CUDA device unavailable; no CPU fallback')
    x, y, hashes = training()
    head = Head()
    output = Path('runs/s309') / f'{args.mode}.json'
    if output.exists():
        raise FileExistsError(output)
    library_sha = sha(os.environ['S309_LIBRARY'])
    records = []
    if args.mode == 'smoke':
        selected = np.concatenate([np.flatnonzero(y == k)[:32] for k in range(3)])
        mini, target = np.ascontiguousarray(x[selected, :8]), np.ascontiguousarray(y[selected])
        old, _ = head.fit(mini, target, 0, legacy=True)
        cpu, info_cpu = head.fit(mini, target, 0)
        gpu, info_gpu = head.fit(mini, target, 1)
        np.testing.assert_array_equal(old, cpu)
        if abs(info_cpu['loss'] - info_gpu['loss']) > 1e-5:
            raise ValueError('small source-matrix CPU/GPU objective mismatch')
        probabilities = torch.softmax(torch.from_numpy(mini @ gpu[:, :-1].T + gpu[:, -1]), 1).numpy()
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1):
            raise ValueError('GPU-fitted posterior invalid')
        records = [info_cpu, info_gpu]
    else:
        gate = json.loads(Path('runs/s309/smoke.json').read_text())
        if gate['library_sha256'] != library_sha or gate['input_sha256'] != hashes or not gate['passed']:
            raise ValueError('source/library smoke gate differs')
        for order in ((0, 1), (1, 0), (0, 1)):
            pair = []
            for device in order:
                _, info = head.fit(x, y, device)
                pair.append(info)
                print(json.dumps(info), flush=True)
            if abs(pair[0]['loss'] - pair[1]['loss']) > 1e-5:
                raise ValueError('full source-matrix CPU/GPU objective mismatch')
            records.extend(pair)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump({'passed': True, 'input_sha256': hashes, 'library_sha256': library_sha,
                   'source_shape': list(x.shape), 'records': records,
                   'cuda_device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                   'cpu_affinity': sorted(os.sched_getaffinity(0)),
                   'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'held_accuracy_computed': False,
                   'scope': 'source-only full solve, includes copies, not first target iterate'}, f, indent=2)
    print(f'passed {args.mode}: {output}', flush=True)


if __name__ == '__main__':
    main()
