"""Check the native binary inside the worker replica; no held-person score."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bridge import Native, Session


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_device(native, train_x, train_y, people, held_x, device):
    reports = []
    for arm in range(4):
        started = time.monotonic()
        solver = native.create(arm, device, 17)
        solver.fit(train_x, train_y, people, held=1, epochs=1, consistency=int(arm == 3))
        frozen = Session(solver)
        adaptive = Session(solver)
        fresh = Session(solver)
        first = None
        for i, window in enumerate(held_x):
            base = frozen.step(window, False)
            live = adaptive.step(window, True)
            if i < 40 and not np.array_equal(base, live):
                raise ValueError(f'arm{arm}: update changed an ineligible window{i}')
            if not np.isfinite(base).all() or not np.isfinite(live).all():
                raise ValueError(f'arm{arm}: invalid native prediction')
            if first is None:
                first = base.copy()
        status = adaptive.solver.status()
        if status['windows'] != 61 or status['updates'] != 2 or not 0 < status['delta'] <= .100001:
            raise ValueError(f'arm{arm}: adaptation did not execute correctly: {status}')
        if fresh.solver.status()['windows'] != 0:
            raise ValueError('native state crossed handles')
        adaptive.reset()
        if not np.array_equal(adaptive.step(held_x[0], True), first):
            raise ValueError('native reset did not restore population predictions')
        if device:
            torch.cuda.synchronize()
        reports.append({'arm': arm, 'seconds': time.monotonic() - started, 'status': status})
        for item in (frozen.solver, adaptive.solver, fresh.solver, solver):
            item.close()
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('worker-replica GPU check requires a visible CUDA device')
    with np.load(args.data, allow_pickle=False) as z:
        subject = z['subject']
        pids = np.array([int(s.split('/')[-1][1:]) for s in subject], np.int32)
        fitting = np.flatnonzero(pids != 1)
        labels = z['y'][fitting]
        chosen = []
        for person in range(2, 11):
            for label in range(3):
                candidates = fitting[(pids[fitting] == person) & (labels == label)]
                if not len(candidates):
                    raise ValueError(f'missing training person/class {person}/{label}')
                chosen.append(candidates[0])
        chosen = np.asarray(chosen)
        x = np.ascontiguousarray(z['X'][chosen], np.float32)
        y = np.ascontiguousarray(z['y'][chosen], np.int64)
        held = np.ascontiguousarray(z['X'][np.flatnonzero(pids == 1)[:61]], np.float32)
    if len(held) != 61 or 1 in pids[chosen]:
        raise ValueError('worker probe fold is invalid')
    native = Native(args.library)
    report = {'torch': torch.__version__, 'torch_file': torch.__file__,
              'python': sys.version, 'cuda_runtime': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'library_sha256': digest(args.library),
              'data_sha256': digest(args.data), 'fit_people': list(range(2, 11)),
              'held_person': 1, 'held_labels_used': False, 'held_accuracy_computed': False}
    for device, name in ((0, 'cpu'), (1, 'cuda')):
        report[name] = check_device(native, x, y, pids[chosen], held, device)
    maps = Path('/proc/self/maps').read_text().splitlines()
    forbidden = ('neural-interfaces-2026/.venv/', 'flightrl-puffer5.pmbvox/toolchain/')
    if any(token in line for line in maps for token in forbidden):
        raise RuntimeError('worker check loaded a host Torch/compiler library')
    report['loaded_numerical_libraries'] = sorted({line.split()[-1] for line in maps
        if '/' in line and any(name in line for name in ('libtorch', 'libstdc++', 'libc10'))})
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print('WORKER_REPLICA_NATIVE_CPU_CUDA_OK', args.output)


if __name__ == '__main__':
    main()
