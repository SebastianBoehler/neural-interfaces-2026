"""Validate §304 fold outputs without reading held-out labels or scores."""
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('plain', 'unconditional10', 'conditional10', 'conditional30')
FAMILIES = ('fb304', 'fb_uc304', 'fb_cc10', 'fb_cc30')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
MANIFEST = '94a5f8a19c3532017f42fc1e85520e9ca0aa511e2ece1bc40df9f406a65b45aa'
DATA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
PERMS = '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def inputs(root, protocol):
    if (sha(root / 'coral304.sha256'), sha(root / 'data/proteus_windows.npz'),
            sha(root / 'runs/shuffle_perms.npz')) != (MANIFEST, DATA, PERMS):
        raise ValueError('source or input manifest mismatch')
    with np.load(root / 'data/proteus_windows.npz', allow_pickle=False) as z:
        d = {k: z[k] for k in ('subject', 'split')}
    if len(d['subject']) != 2655:
        raise ValueError('unexpected dataset rows')
    folds = {held: (train, test) for held, train, test in protocol.folds(d, 'all_loso')}
    if set(folds) != {f'P{i:02d}' for i in range(1, 11)}:
        raise ValueError('unexpected held-person folds')
    with np.load(root / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    if any(not np.array_equal(np.sort(p), np.arange(2655)) for p in perms.values()):
        raise ValueError('invalid pinned order')
    return d, folds, perms


def fold(root, d, folds, perms, arm, seed, held):
    path = root / f'runs/coral304/{arm}_s{seed}_{held}.npz'
    before = sha(path); train, test = folds[held]
    fit = sorted(set(d['subject'][train]))
    if len(fit) != 9 or set(fit) & set(d['subject'][test]):
        raise ValueError(f'{path.name}: incorrect fold')
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'metadata'} | {f'{o}|{k}' for o in ORDERS for k in ('index', 'proba')}:
            raise ValueError(f'{path.name}: output keys')
        m = json.loads(str(z['metadata'].item()))
        if ((m['arm'], m['seed'], m['held']) != (arm, seed, held) or
                m['fit_people'] != fit or m['manifest_sha256'] != MANIFEST or
                m['data_sha256'] != DATA or m['permutation_sha256'] != PERMS or
                m['train_rows_sha256'] != hashlib.sha256(np.flatnonzero(train).astype(np.int64).tobytes()).hexdigest() or
                len(m['parameter_sha256']) != 64):
            raise ValueError(f'{path.name}: provenance mismatch')
        if arm != 'plain' and (m['dg_stats']['coral_pairs'] <= 0 or m['dg_stats']['coral_pen'] <= 0):
            raise ValueError(f'{path.name}: inactive penalty')
        if arm.startswith('conditional'):
            p = m['penalty']
            if (p['batches'] <= 0 or min(p['class_pairs']) <= 0 or
                    not np.isfinite(p['gradient_norm']) or p['gradient_norm'] <= 0):
                raise ValueError(f'{path.name}: inactive class penalty')
        values = {}
        for order, perm in perms.items():
            rows = z[f'{order}|index']; probs = z[f'{order}|proba']
            expected = perm[test[perm]]
            if (not np.array_equal(rows, expected) or probs.shape != (len(rows), 3) or
                    not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or
                    not np.allclose(probs.sum(1), 1, atol=1e-12, rtol=0)):
                raise ValueError(f'{path.name}/{order}: invalid rows or probabilities')
            values[order] = (rows.copy(), probs.copy())
    if sha(path) != before:
        raise ValueError(f'{path.name}: changed while reading')
    return before, values
