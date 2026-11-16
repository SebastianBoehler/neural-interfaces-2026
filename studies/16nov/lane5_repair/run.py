"""Replay one §293 checkpoint under fixed §303 policies; never load labels."""
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path.insert(0, str(ROOT / 'studies/16nov/lane5'))
from bridge import Native, Session  # noqa: E402
from run import ARMS, ORDERS, orders, probability, sha, task_config  # noqa: E402
import protocol  # noqa: E402

spec = importlib.util.spec_from_file_location('repair_bridge', ROOT / 'studies/16nov/lane5_repair/bridge.py')
repair_bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(repair_bridge)
MODES = ('static', 'adaptive', 'repair', 'reverse')


def check_status(status, n):
    attempts = len(range(40, n, 20)); counts = status['selections']
    if (status['windows'] != n or status['attempts'] != attempts or len(counts) != attempts or
            status['updates'] + status['skips'] != attempts or sum(counts) != status['selected_total'] or
            status['updates'] != sum(x >= 8 for x in counts) or
            any(x < 0 or x > min(64, 40 + 20 * i) for i, x in enumerate(counts)) or
            not np.isfinite([status['max_grad'], status['delta']]).all() or status['delta'] > .030001 or
            (status['updates'] and (status['max_grad'] <= 0 or status['delta'] <= 0)) or
            (not status['updates'] and status['delta'] != 0)):
        raise ValueError(f'invalid repair status: {status}')


def stream(population, extension, windows):
    static, adaptive = Session(population), Session(population)
    repair, reverse = extension.create(population, 1), extension.create(population, 2)
    before = population.params().copy(); teacher = repair.teacher(windows[0])
    result = {k: np.empty((len(windows), 3), np.float64) for k in MODES}
    for i, x in enumerate(windows):
        result['static'][i] = probability(static.step(x, False))
        result['adaptive'][i] = probability(adaptive.step(x, True))
        result['repair'][i] = probability(repair.step(x))
        result['reverse'][i] = probability(reverse.step(x))
    if any(not np.allclose(result['static'][:40], result[k][:40], atol=1e-7, rtol=0)
           for k in ('repair', 'reverse')):
        raise ValueError('repair changed predictions before first update')
    if not np.array_equal(before, population.params()) or not np.array_equal(teacher, repair.teacher(windows[0])):
        raise ValueError('population or frozen teacher changed')
    status = {'repair': repair.status(), 'reverse': reverse.status()}
    for x in status.values():
        check_status(x, len(windows))
    static.solver.close(); adaptive.solver.close(); repair.close(); reverse.close()
    return result, status


def verify_legacy(actual, reference, name):
    if actual.shape != reference.shape or not np.allclose(actual, reference, atol=2e-5, rtol=0):
        raise ValueError(f'legacy probability mismatch: {name}')
    margin = np.sort(reference, axis=1)[:, -1] - np.sort(reference, axis=1)[:, -2]
    safe = margin > 4e-5
    if np.any(actual[safe].argmax(1) != reference[safe].argmax(1)):
        raise ValueError(f'legacy non-tied label mismatch: {name}')
    return {'max_abs': float(np.max(np.abs(actual - reference))), 'near_ties': int((~safe).sum())}


def main(task):
    (arm, architecture, _), seed, index = task_config(task)
    manifest = Path(os.environ['REPAIR_MANIFEST']).resolve(strict=True)
    library = Path(os.environ['LANE5_LIB']).resolve(strict=True)
    extension_path = Path(os.environ['LANE5_REPAIR_LIB']).resolve(strict=True)
    if sha(manifest) != os.environ['REPAIR_MANIFEST_SHA']:
        raise ValueError('repair manifest lock mismatch')
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in ('X', 'subject', 'session', 'split')}
    d['stream'] = np.char.add(np.char.add(d['subject'], '|'), d['session'])
    folds = list(protocol.folds(d, 'all_loso'))
    if len(folds) != 10:
        raise ValueError('expected ten held-out people')
    held, train, test = folds[index]
    checkpoint = ROOT / f'runs/lane5_checkpoints/{arm}_s{seed}_{held}.pt'
    stored = ROOT / f'runs/lane5/{arm}_s{seed}_{held}.npz'
    with (ROOT / 'frozen/records.json').open() as f:
        records = json.load(f)
    if sha(checkpoint) != records[checkpoint.name] or sha(stored) != records[stored.name]:
        raise ValueError('checkpoint or stored reference hash mismatch')
    output = ROOT / f'runs/repair/{arm}_s{seed}_{held}.npz'
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU requested but unavailable')
    gpu_name = torch.cuda.get_device_name(0)
    if 'A100' not in gpu_name:
        raise RuntimeError(f'fixed precision route requires A100, got {gpu_name}')
    with np.load(stored, allow_pickle=False) as z:
        source_meta = json.loads(str(z['metadata'].item()))
    original_gpu = source_meta['gpu_name']
    flags_before = {'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                    'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                    'cudnn_benchmark': torch.backends.cudnn.benchmark,
                    'cudnn_deterministic': torch.backends.cudnn.deterministic,
                    'deterministic': torch.are_deterministic_algorithms_enabled()}
    if 'RTX 2080 Ti' in original_gpu:
        precision_policy = 'original_2080_a100_cudnn_fp32'
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    elif 'A100' in original_gpu:
        precision_policy = 'original_a100_default'
        if not flags_before['cudnn_tf32'] or flags_before['matmul_tf32']:
            raise RuntimeError(f'A100 runtime defaults differ: {flags_before}')
    else:
        raise ValueError(f'unknown original device: {original_gpu}')
    flags_after = {'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                   'matmul_tf32': torch.backends.cuda.matmul.allow_tf32}
    native = Native(library); extension = repair_bridge.RepairNative(extension_path)
    population = native.create(architecture, 1, seed); population.load(checkpoint)
    permutations = orders(ROOT, len(d['subject']))
    result, logs, order_hash, legacy_checks = {}, {}, {}, {}
    t0 = time.monotonic()
    with np.load(stored, allow_pickle=False) as reference:
        oldmeta = json.loads(str(reference['metadata'].item()))
        if (oldmeta['arm'], oldmeta['seed'], oldmeta['held']) != (arm, seed, held):
            raise ValueError('stored checkpoint identity mismatch')
        if set(oldmeta['fit_people']) != set(d['subject'][train]):
            raise ValueError('fit people differ from held-out fold')
        for order, perm in permutations.items():
            rows = perm[test[perm]]
            if not np.array_equal(reference[f'{order}|index'], rows):
                raise ValueError(f'stored order differs: {order}')
            order_hash[order] = hashlib.sha256(perm.tobytes()).hexdigest()
            block = {k: np.empty((len(rows), 3), np.float64) for k in MODES}
            entries = []
            for session in dict.fromkeys(d['stream'][rows]):
                loc = np.flatnonzero(d['stream'][rows] == session)
                values, status = stream(population, extension, d['X'][rows[loc]])
                for k in MODES:
                    block[k][loc] = values[k]
                entries.append({'session': str(session), **status})
            legacy_checks[order] = {k: verify_legacy(block[k], reference[f'{order}|{k}'], f'{order}/{k}')
                                    for k in ('static', 'adaptive')}
            result[f'{order}|index'] = rows
            for k in MODES:
                result[f'{order}|{k}'] = block[k]
            logs[order] = entries
    metadata = {'arm': arm, 'architecture': architecture, 'seed': seed, 'held': held,
                'fit_people': sorted(set(d['subject'][train])), 'orders': list(ORDERS), 'order_sha': order_hash,
                'status': logs, 'legacy_checks': legacy_checks, 'checkpoint_sha256': sha(checkpoint),
                'reference_sha256': sha(stored), 'manifest_sha256': sha(manifest),
                'library_sha256': sha(library), 'extension_sha256': sha(extension_path),
                'data_sha256': sha(ROOT / 'data/proteus_windows.npz'),
                'permutations_sha256': sha(ROOT / 'runs/shuffle_perms.npz'),
                'gpu_name': gpu_name, 'original_gpu': original_gpu, 'precision_policy': precision_policy,
                'flags_before': flags_before, 'flags_after': flags_after, 'torch_version': str(torch.__version__),
                'torch_cuda': torch.version.cuda, 'elapsed_s': time.monotonic() - t0}
    result['metadata'] = json.dumps(metadata, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f, **result)
    population.close()
    print(f'saved {output}; no labels or metrics read', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: run.py TASK')
    main(int(sys.argv[1]))
