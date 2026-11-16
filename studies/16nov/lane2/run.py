"""Generate fold predictions and provenance only; never compute a treatment score.

Task 0..29 maps fold = task % 10, seed = task // 10. Run on Galvani only.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(os.environ.get('NI26_WS', Path(__file__).resolve().parents[3])).resolve()
sys.path[:0] = [str(ROOT / 'studies/16nov/lane2'), str(ROOT / 'experiments')]
import protocol  # noqa: E402
from model import features, train_population  # noqa: E402
from stream import predict_session  # noqa: E402

ORDERS = tuple(f'{name}{i}' for name in ('within', 'full', 'runs') for i in (0, 1))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ranks(perms, order, n):
    scheme = order.rstrip('0123456789')
    key = f'{scheme}|{order[len(scheme):]}|all_loso'
    p = np.asarray(perms[key], dtype=np.int64)
    if not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError(f'{key}: not a permutation of global indices')
    rank = np.empty(n, dtype=np.int64); rank[p] = np.arange(n)
    return rank


def main(task):
    if not 0 <= task < 30:
        raise ValueError('task must be in 0..29')
    d = protocol.load(); folds = list(protocol.folds(d, 'all_loso'))
    if len(folds) != 10:
        raise ValueError('expected ten LOSO folds')
    person, tr, te = folds[task % 10]; seed = task // 10
    rows = np.flatnonzero(te); it = np.flatnonzero(tr)
    if set(d['subject'][it]) & set(d['subject'][rows]):
        raise ValueError('held-out person appears in train')
    if len(rows) + len(it) != len(d['y']):
        raise ValueError('fold does not cover dataset')
    out = ROOT / 'runs/lane2' / f'fold_{person}_s{seed}.npz'
    if out.exists():
        raise FileExistsError(out)
    data_path = ROOT / 'data/proteus_windows.npz'
    perm_path = ROOT / 'runs/shuffle_perms.npz'
    source = [Path(__file__).resolve(), ROOT / 'studies/16nov/lane2/model.py',
              ROOT / 'studies/16nov/lane2/stream.py', ROOT / 'experiments/protocol.py']
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in [data_path, perm_path, *source]}
    started = time.monotonic(); Xtrain = features(d['X'][it])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m, mu, sd = train_population(Xtrain, d['y'][it], d['subject'][it], seed, device)
    del Xtrain
    perms = np.load(perm_path)
    data = {'index': rows, 'person': np.asarray(person), 'seed': np.asarray(seed),
            'manifest': np.asarray(json.dumps(manifest, sort_keys=True))}
    positions = np.full(len(d['y']), -1, dtype=np.int64); positions[rows] = np.arange(len(rows))
    all_diagnostics = {}; sessions = sorted(set(d['stream'][rows]))
    for order in ORDERS:
        rank = ranks(perms, order, len(d['y']))
        static = np.full((len(rows), 3), np.nan, dtype=np.float32)
        adaptive = static.copy(); off = static.copy()
        diagnostics = []; off_sec = adapt_sec = 0.0
        for session_number, session in enumerate(sessions):
            srows = rows[d['stream'][rows] == session]
            srows = srows[np.argsort(rank[srows])]
            x_session = np.clip((features(d['X'][srows]) - mu) / sd, -8, 8).astype(np.float32)
            t = time.monotonic()
            base, disabled, _ = predict_session(m, x_session, seed + session_number, update=False)
            if device == 'cuda': torch.cuda.synchronize()
            off_sec += time.monotonic() - t
            t = time.monotonic()
            check, live, diag = predict_session(m, x_session, seed + session_number, update=True)
            if device == 'cuda': torch.cuda.synchronize()
            adapt_sec += time.monotonic() - t
            if not np.array_equal(base, disabled) or not np.array_equal(base, check):
                raise AssertionError('static and update-off paths differ')
            if not np.array_equal(base[:20], live[:20]):
                raise AssertionError('adaptation changed pre-update windows')
            loc = positions[srows]
            static[loc], off[loc], adaptive[loc] = base, disabled, live
            diagnostics.append({'session': session, **diag})
        for name, values in (('static', static), ('off', off), ('adaptive', adaptive)):
            if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1, atol=1e-6):
                raise AssertionError(f'{order}/{name}: invalid outputs')
            data[f'{name}|{order}'] = values
        all_diagnostics[order] = {'sessions': diagnostics, 'off_seconds': off_sec,
                                  'adapt_seconds': adapt_sec, 'windows': len(rows)}
        print(f'{person} s{seed} {order}: selected={sum(sum(x["selected"]) for x in diagnostics)} '
              f'off={1000*off_sec/len(rows):.2f}ms/window adapt={1000*adapt_sec/len(rows):.2f}ms/window', flush=True)
    data['diagnostics'] = np.asarray(json.dumps(all_diagnostics, sort_keys=True))
    data['device'] = np.asarray(torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu')
    data['peak_allocated_bytes'] = np.asarray(torch.cuda.max_memory_allocated() if device == 'cuda' else -1)
    data['elapsed_seconds'] = np.asarray(time.monotonic() - started)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as f:
        np.savez_compressed(f, **data)
    print(f'saved {out}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python studies/16nov/lane2/run.py TASK')
    main(int(sys.argv[1]))
