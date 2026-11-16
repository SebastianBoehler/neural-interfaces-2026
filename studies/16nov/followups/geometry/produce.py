"""§295: one person, all eight orders; save unscored posterior deltas."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
os.environ['NI26_ROOT'] = str(ROOT)
sys.path[:0] = [str(ROOT / 'studies/16nov/followups/geometry'), str(ROOT / 'experiments')]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
from chain import ORDERS  # noqa: E402
from config import ARMS, ARCHIVES  # noqa: E402
from diag import stream  # noqa: E402
from pilot import sha, vectorize, whitened  # noqa: E402


def main(task):
    if not 0 <= task < 10 or not torch.cuda.is_available():
        raise ValueError('task 0..9 and an available GPU are required')
    d = protocol.load(); folds = list(protocol.folds(d, 'all_loso'))
    if len(folds) != 10:
        raise ValueError('expected ten person-disjoint folds')
    held, tr, te = folds[task]; fit = np.flatnonzero(tr); index = np.flatnonzero(te)
    people = sorted(set(d['subject'][tr]))
    if len(people) != 9 or set(people) & set(d['subject'][te]):
        raise ValueError('invalid held-person partition')
    output = ROOT / f'runs/s295_geometry/shards/{held}.npz'
    if output.exists():
        raise FileExistsError(output)
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        orders = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    for name, perm in orders.items():
        if not np.array_equal(np.sort(perm), np.arange(len(d['subject']))):
            raise ValueError(f'invalid permutation {name}')
    started = time.monotonic()
    fe = T.Feats(d, T.parse('car_ov17_fs2_t0.1_le_k2'))
    model = T.fit_head('lr0.3', fe.F[fit], d['y'][fit], fe.slices)
    scaler, lr = model.steps[0][1], model.steps[1][1]
    if not np.array_equal(lr.classes_, [0, 1, 2]):
        raise ValueError('core classifier class order mismatch')
    size = 51 * 41 * 42 // 2
    if len(fe.C) != 51 or lr.coef_.shape[1] < size:
        raise ValueError('unexpected core feature dimensions')
    head = (lr.coef_[:, :size] / scaler.scale_[:size]).reshape(3, 51, -1)
    model_digest = hashlib.sha256(b''.join(np.ascontiguousarray(x).tobytes() for x in
        (scaler.mean_, scaler.scale_, lr.coef_, lr.intercept_))).hexdigest()
    gpu_head = torch.from_numpy(head).to('cuda')
    data = {'index': index}; records = {}; max_diff = 0.; max_feature = 0.
    for order, perm in orders.items():
        values = {arm: np.full((len(index), 3), np.nan) for arm in ARMS}
        rows = perm[te[perm]]
        archive = ROOT / 'runs' / ARCHIVES[order]
        with np.load(archive, allow_pickle=False) as z:
            expected = z[f'{order}|all_loso']
        for session in dict.fromkeys(d['stream'][rows]):
            r = rows[d['stream'][rows] == session]
            w = whitened(fe, r, fit)
            n = np.arange(1, len(r) + 1)[:, None]
            slow = fe.M[r] - np.cumsum(fe.M[r], axis=0) / (fe.k + n)
            features = vectorize(w, slow)
            partial = fe.causal(r[:10], [fe.prior(b, fit) for b in range(51)],
                                [z[fit].mean() for z in fe.d2], None)
            feature_diff = float(np.max(np.abs(features[:10] - partial)))
            rebuilt = model.predict_proba(features)
            diff = float(np.max(np.abs(rebuilt - expected[r])))
            if feature_diff > 1e-8 or diff > 1e-5 or np.any(rebuilt.argmax(1) != expected[r].argmax(1)):
                raise ValueError(f'{held}/{order}/{session}: frozen-source parity failed: {feature_diff=} {diff=}')
            max_diff, max_feature = max(max_diff, diff), max(max_feature, feature_diff)
            matrix = torch.from_numpy(w).to('cuda')
            local = np.searchsorted(index, r)
            for arm, (mode, cap) in ARMS.items():
                before = time.monotonic()
                delta, status = stream(matrix, rebuilt, gpu_head, mode, cap)
                torch.cuda.synchronize()
                if not np.array_equal(delta[:32], np.zeros_like(delta[:32])) or not np.allclose(delta.sum(1), 0, atol=1e-12):
                    raise ValueError(f'{held}/{order}/{arm}: invalid causal posterior delta')
                values[arm][local] = delta
                records.setdefault(f'{arm}|{order}', []).append({'session': str(session),
                    'windows': len(r), 'seconds': time.monotonic() - before, **status})
            del matrix, w
        for arm in ARMS:
            if not np.isfinite(values[arm]).all():
                raise ValueError(f'{held}/{order}/{arm}: incomplete rows')
            data[f'{arm}|{order}'] = values[arm]
        print(f'{held} {order}: {len(rows)} windows generated, no score', flush=True)
    metadata = {'held': held, 'fit_people': people, 'fit_rows_sha256': hashlib.sha256(fit.tobytes()).hexdigest(),
                'model_sha256': model_digest, 'max_rebuilt_diff': max_diff, 'max_feature_diff': max_feature,
                'data_sha256': sha(ROOT / 'data/proteus_windows.npz'), 'permutation_sha256': sha(ROOT / 'runs/shuffle_perms.npz'),
                'archives_sha256': {name: sha(ROOT / 'runs' / name) for name in sorted(set(ARCHIVES.values()))},
                'source_manifest_sha256': sha(ROOT / 'source.sha256'), 'records': records,
                'order_sha256': {o: hashlib.sha256(p.tobytes()).hexdigest() for o, p in orders.items()},
                'elapsed_s': time.monotonic() - started, 'held_accuracy_computed': False}
    data['metadata'] = json.dumps(metadata, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f, **data)
    print(f'saved {output}; no score', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: produce.py FOLD_INDEX')
    main(int(sys.argv[1]))
