"""§295 single-session, no-score pilot of the reconstructed-core delta adapter."""
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
from diag import logits_delta, stream  # noqa: E402
from spdlog import spd_log  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def whitened(fe, rows, fit_rows):
    """Existing §175 causal reference; each matrix includes current-window covariance."""
    n = np.arange(1, len(rows) + 1)[:, None, None]
    blocks = np.empty((len(rows), len(fe.C), fe.C[0].shape[1], fe.C[0].shape[2]), np.float64)
    for b in range(len(fe.C)):
        meanlog = (fe.k * fe.prior(b, fit_rows) + np.cumsum(fe.L[b][rows], axis=0)) / (fe.k + n)
        p = T.FT.invsqrtm(T.FT.expm_sym(meanlog))
        blocks[:, b] = p @ fe.C[b][rows] @ p.transpose(0, 2, 1)
    return blocks


def vectorize(blocks, slow):
    parts = [T.FT.upper(T.FT.logm(blocks[:, b])) for b in range(blocks.shape[1])]
    return np.concatenate([*parts, slow], axis=1)


def check_gradient(blocks, head):
    samples = torch.from_numpy(blocks[:2, :2].copy())
    weights = torch.from_numpy(head[:, :2].copy())
    bias = torch.linspace(-.02, .02, samples.shape[-1], dtype=torch.float64)
    results = []
    for device in ('cpu', 'cuda'):
        x, h = samples.to(device), weights.to(device)
        a = bias.to(device).requires_grad_()
        reference = spd_log(x).detach()
        y = logits_delta(x, reference, h, a)
        loss = y.square().mean(); grad, = torch.autograd.grad(loss, a)
        if not torch.isfinite(y).all() or not torch.isfinite(grad).all() or grad.norm().item() <= 0:
            raise ValueError(f'{device}: invalid or inactive real covariance gradient')
        results.append((y.detach().cpu().numpy(), grad.detach().cpu().numpy()))
    dlog = float(np.max(np.abs(results[0][0] - results[1][0])))
    dgrad = float(np.max(np.abs(results[0][1] - results[1][1])))
    if dlog > 1e-6 or dgrad > 1e-6:
        raise ValueError(f'CPU/GPU geometry mismatch: logits={dlog} gradients={dgrad}')
    return {'logit_max_abs': dlog, 'gradient_max_abs': dgrad}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('pilot requires a Galvani GPU')
    started = time.monotonic()
    d = protocol.load(); folds = list(protocol.folds(d, 'all_loso'))
    held, tr, te = folds[0]
    if held != 'P01' or set(d['subject'][tr]) & set(d['subject'][te]):
        raise ValueError('held-person exclusion failed')
    fit_rows = np.flatnonzero(tr)
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        order = z['within|0|all_loso'].astype(np.int64)
    if not np.array_equal(np.sort(order), np.arange(len(d['subject']))):
        raise ValueError('invalid within0 permutation')
    rows = order[te[order]]
    session = d['stream'][rows[0]]
    rows = rows[d['stream'][rows] == session]
    if len(rows) < 72:
        raise ValueError('pilot session too short')
    cfg = T.parse('car_ov17_fs2_t0.1_le_k2')
    fe = T.Feats(d, cfg)
    model = T.fit_head('lr0.3', fe.F[fit_rows], d['y'][fit_rows], fe.slices)
    scaler, lr = model.steps[0][1], model.steps[1][1]
    if not np.array_equal(lr.classes_, [0, 1, 2]):
        raise ValueError('core head class order differs')
    start = time.monotonic(); blocks = whitened(fe, rows, fit_rows)
    n = np.arange(1, len(rows) + 1)[:, None]
    slow = fe.M[rows] - np.cumsum(fe.M[rows], axis=0) / (fe.k + n)
    X = vectorize(blocks, slow)
    check = fe.causal(rows[:10], [fe.prior(b, fit_rows) for b in range(len(fe.C))],
                      [z[fit_rows].mean() for z in fe.d2], None)
    feature_diff = float(np.max(np.abs(X[:10] - check)))
    if feature_diff > 1e-8:
        raise ValueError(f'whitened/prefix feature parity failed: {feature_diff}')
    rebuilt = model.predict_proba(X)
    archive = ROOT / 'runs/coremember.npz'
    with np.load(archive, allow_pickle=False) as z:
        saved = z['within0|all_loso'][rows]
    diff = float(np.max(np.abs(rebuilt - saved)))
    labels = int(np.count_nonzero(rebuilt.argmax(1) != saved.argmax(1)))
    if diff > 1e-5 or labels:
        raise ValueError(f'historical rebuilt-core mismatch: {diff=} label_changes={labels}')
    size = 51 * (41 * 42 // 2)
    if len(fe.C) != 51 or X.shape[1] < size:
        raise ValueError('unexpected core feature block shape')
    head = (lr.coef_[:, :size] / scaler.scale_[:size]).reshape(3, 51, -1)
    gradient = check_gradient(blocks, head)
    prep_s = time.monotonic() - start
    transfer_t0 = time.monotonic()
    matrices = torch.from_numpy(blocks).to('cuda')
    w = torch.from_numpy(head).to('cuda')
    torch.cuda.synchronize()
    transfer_s = time.monotonic() - transfer_t0
    short, _ = stream(matrices[:40], rebuilt[:40], w, 'diag', .5)
    stream(matrices[:40], rebuilt[:40], w, 'scalar', .5)
    repeated, _ = stream(matrices[:40], rebuilt[:40], w, 'diag', .5)
    if not np.allclose(short, repeated, atol=1e-8, rtol=1e-8):
        raise ValueError('geometry reset/interleaving changed prefix')
    result = {}
    for mode, cap in (('zero', 0.), ('scalar', .5), ('diag', .5)):
        torch.cuda.synchronize(); t0 = time.monotonic()
        delta, status = stream(matrices, rebuilt, w, mode, cap)
        torch.cuda.synchronize(); elapsed = time.monotonic() - t0
        if (not np.array_equal(delta[:32], np.zeros_like(delta[:32])) or
                not np.allclose(delta.sum(axis=1), 0, atol=1e-12) or
                (mode == 'zero' and not np.array_equal(delta, np.zeros_like(delta)))):
            raise ValueError(f'{mode}: invalid zero/prefix/posterior delta')
        if mode == 'diag' and not np.allclose(delta[:40], short, atol=1e-8, rtol=1e-8):
            raise ValueError('future windows changed geometry prefix')
        result[mode] = {'elapsed_s': elapsed, **status}
    print(json.dumps({'held': held, 'order': 'within0', 'session': str(session), 'windows': len(rows),
                      'fit_people': sorted(set(d['subject'][tr])), 'fit_rows': len(fit_rows),
                      'feature_prefix_diff': feature_diff, 'rebuilt_core_max_diff': diff,
                      'rebuilt_label_changes': labels, 'cpu_gpu_geometry': gradient,
                      'data_sha256': sha(ROOT / 'data/proteus_windows.npz'),
                      'permutation_sha256': sha(ROOT / 'runs/shuffle_perms.npz'),
                      'core_archive_sha256': sha(archive), 'preparation_s': prep_s, 'transfer_s': transfer_s,
                      'total_s': time.monotonic() - started, 'modes': result,
                      'held_accuracy_computed': False}, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
