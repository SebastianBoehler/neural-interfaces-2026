"""Unscored, small real-data implementation smoke; no outer-person labels are read."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ['S308_BASE']).resolve(strict=True)
CACHE = Path(os.environ['S308_CACHE']).resolve(strict=True)
sys.path[:0] = [str(HERE), str(BASE / 'experiments'), str(BASE / 'studies/16nov/followups/native_core')]
import protocol  # noqa: E402
from causal_native import CausalNative  # noqa: E402
from export import ridge_export, structured_export  # noqa: E402
from geometry import source_prior, sessions_in_order  # noqa: E402
from native import NativeHead  # noqa: E402
from heads import ridge_predict  # noqa: E402
from split import source_partition  # noqa: E402
from train import fit_ridge_grid, fit_structured, refit_ridge, refit_structured  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path(os.environ['S308_SMOKE_OUT'])
    out.mkdir(parents=True, exist_ok=False)
    d = protocol.load()
    people = np.array([int(s.rsplit('P', 1)[1]) - 1 for s in d['subject']])
    train_mask, inner_mask, outer_mask = source_partition(people, outer_person=0, seed=0)
    if set(people[train_mask]) != set(range(2, 10)) or set(people[inner_mask]) != {1}:
        raise ValueError('unexpected person selection')
    hashes, priors, counts, x_parts, y_parts, inner = {}, [], [], [], [], None
    for person in range(10):
        path = CACHE / f'P{person + 1:02d}.npz'
        hashes[path.name] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            rows = z['rows'].copy()
            if meta['person'] != f'P{person + 1:02d}' or not np.array_equal(rows, np.flatnonzero(people == person)):
                raise ValueError('cached person row provenance mismatch')
            if meta['held_accuracy_computed'] is not False or meta['rows_sha256'] != hashlib.sha256(rows.tobytes()).hexdigest():
                raise ValueError('invalid cache metadata')
            if meta['native_core_sha256'] != digest(Path(os.environ['NATIVE_CORE_LIB'])) or meta['native_train_sha256'] != digest(Path(os.environ['NATIVE_TRAIN_LIB'])):
                raise ValueError('cache was created by another native library')
            priors.append(z['prior'].copy())
            counts.append(len(rows))
            if person == 1:
                inner = {'rows': rows, 'cov': z['cov'].copy(), 'slow': z['slow'].copy()}
            elif person >= 2:
                # Only a tiny balanced source subset; this is not a production model.
                chosen = np.concatenate([np.flatnonzero(d['y'][rows] == c)[:2] for c in range(3)])
                if len(chosen) != 6:
                    raise ValueError('insufficient source-class rows for smoke')
                x_parts.append(z['features'][chosen].copy())
                y_parts.append(d['y'][rows[chosen]].copy())
    x, y = np.concatenate(x_parts), np.concatenate(y_parts)
    priors, counts = np.asarray(priors), np.asarray(counts)
    device = int(os.environ['S308_DEVICE'])
    native = CausalNative(device)
    torch_device = 'cuda' if device == 1 else 'cpu'
    nine = source_prior(priors, counts, [0])
    if not np.allclose(nine, native.combine(np.ascontiguousarray(priors), counts, 0), atol=1e-12, rtol=1e-12):
        raise ValueError('population-prior formula disagrees with native reference')
    eight = source_prior(priors, counts, [0, 1])
    with np.load(BASE / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perm = z['within|0|all_loso'].astype(np.int64)
    groups = sessions_in_order(perm, inner['rows'], d['stream'])
    rows = groups[0][:25]
    if len(rows) < 10 or not np.all(inner_mask[rows]) or np.any(outer_mask[rows]):
        raise ValueError('invalid inner-person smoke session')
    local = np.searchsorted(inner['rows'], rows)
    cov, slow = np.ascontiguousarray(inner['cov'][local]), np.ascontiguousarray(inner['slow'][local])
    valid = native.causal(eight, cov, slow)
    modified = cov.copy()
    modified[-1] = cov[0]
    changed = native.causal(eight, modified, slow)
    reset = native.causal(eight, cov, slow)
    prefix_error = float(np.max(np.abs(valid[:-1] - changed[:-1])))
    reset_error = float(np.max(np.abs(valid - reset)))
    if prefix_error > 1e-10 or reset_error > 1e-10 or not np.isfinite(valid).all():
        raise ValueError('causal feature prefix or reset failed')
    results = {}
    for arm in ('linear_structured', 'tanh_structured', 'linear', 'quadratic', 'fourier'):
        started = time.monotonic()
        path = out / f'{arm}.pt'
        if arm.endswith('_structured'):
            nonlinear = arm == 'tanh_structured'
            chosen, _, epoch, _ = fit_structured(x, y, valid, d['y'][rows], nonlinear,
                                                  0, epochs=(1, 2), batch=48, device=torch_device)
            model, scaler = refit_structured(x, y, nonlinear, 0, epoch, batch=48, device=torch_device)
            if any(not torch.allclose(a, b, atol=2e-5, rtol=2e-5) for a, b in zip(chosen.parameters(), model.parameters())):
                raise ValueError('selected and fixed-epoch source refit differ')
            activity = min(float(model.block.weight.grad.norm()), float(model.slow.weight.grad.norm()))
            structured_export(model, *scaler, valid[:1], path)
            selection = {'epoch': epoch}
            with torch.no_grad():
                test = torch.as_tensor((valid[:8] - scaler[0]) / scaler[1], dtype=torch.float32, device=torch_device)
                direct = model(test).softmax(1).cpu().numpy()
        else:
            chosen = fit_ridge_grid(x, y, valid, d['y'][rows], arm, 0)
            model = refit_ridge(x, y, arm, 0, chosen['strength'], chosen['temperature'])
            if not np.allclose(chosen['weights'], model['weights'], atol=1e-10, rtol=1e-10):
                raise ValueError('selected and fixed-strength source refit differ')
            activity = float(np.linalg.norm(model['weights'][:-1]))
            ridge_export(model, valid[:1], path)
            selection = {'strength': model['strength'], 'temperature': model['temperature']}
            x_valid = np.asarray((valid[:8] - model['mean']) / model['scale'], dtype=np.float32)
            direct = ridge_predict(model['mapper'].transform(x_valid), model['weights'], model['temperature'])
        if not np.isfinite(activity) or activity <= 0:
            raise ValueError(f'{arm}: inactive training gradient or ridge solution')
        expected = torch.jit.load(str(path))(torch.from_numpy(valid[:8])).detach().numpy()
        reference_error = float(np.max(np.abs(expected - direct)))
        if reference_error > 2e-5:
            raise ValueError(f'{arm}: exported checkpoint differs from fitted Python head')
        errors = {}
        for check_device in ((0, 1) if device == 1 else (0,)):
            head = NativeHead(os.environ['S308_LIB'], path, check_device)
            try:
                predicted = head.predict(valid[:8])
            finally:
                head.close()
            errors[str(check_device)] = float(np.max(np.abs(expected - predicted)))
        if max(errors.values()) > 2e-5:
            raise ValueError(f'{arm}: Python/native CPU/GPU output mismatch')
        results[arm] = {'seconds': time.monotonic() - started, 'max_probability_difference': errors,
                        'reference_error': reference_error, 'activity': activity,
                        'checkpoint_sha256': digest(path), 'selection': selection}
    report = {'no_held_accuracy': True, 'outer_person': 'P01', 'inner_person': 'P02',
              'source_people': [f'P{i:02d}' for i in range(3, 11)], 'source_rows': len(x),
              'inner_windows': len(rows), 'input_hashes': hashes, 'prefix_max_abs': prefix_error,
              'reset_max_abs': reset_error, 'library_sha256': digest(Path(os.environ['S308_LIB'])),
              'device': device, 'torch_version': torch.__version__,
              'cuda_name': torch.cuda.get_device_name() if device == 1 else None,
              'cuda_max_allocated': torch.cuda.max_memory_allocated() if device == 1 else None,
              'arms': results}
    (out / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    print('§308 real-data smoke completed; no held accuracy calculated', flush=True)


if __name__ == '__main__':
    main()
