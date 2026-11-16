"""Fit one registered family/seed once per held-out person, stream eight orders."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

FAMILIES = ('eegn', 'eegn_dist2', 'fb_coral3', 'fb_coral10')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')


def mapping(task):
    if not isinstance(task, int) or not 0 <= task < 24:
        raise ValueError(f'invalid task {task}')
    return FAMILIES[task // 6], task % 6


def ordered_rows(rows, ranks):
    rows, ranks = np.asarray(rows), np.asarray(ranks)
    if rows.ndim != 1 or rows.shape != ranks.shape or len(np.unique(rows)) != len(rows):
        raise ValueError('invalid session rows')
    if np.any(ranks < 0) or len(np.unique(ranks)) != len(ranks):
        raise ValueError('invalid or repeated ranks')
    return rows[np.argsort(ranks, kind='stable')].tolist()


def load_module(root):
    spec = importlib.util.spec_from_file_location('ttt7_lane1', root / 'studies/13nov/ttt7.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.VARS, module.DRAW, module.CAP = {}, 0, 0
    return module


def permutations(root, n):
    result = {}
    with np.load(root / 'runs/shuffle_perms.npz') as z:
        for name in ORDERS:
            scheme, seed = name.rstrip('0123456789'), int(name[-1])
            order = z[f'{scheme}|{seed}|all_loso']
            if not np.array_equal(np.sort(order), np.arange(n)):
                raise ValueError(f'invalid permutation {name}')
            ranks = np.empty(n, dtype=int); ranks[order] = np.arange(n)
            result[name] = ranks
    return result


def fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(model.net.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def generate(root, family, seed, smoke=False):
    import torch
    t7 = load_module(root); t5 = t7.T5; d = t7.protocol.load()
    n = len(d['y']); ranks = permutations(root, n)
    out = root / ('runs/diag/ttt/lane1_smoke' if smoke else 'runs/diag/ttt/probs18')
    out.mkdir(parents=True, exist_ok=True)
    paths = {o: out / f'ttt18{o}_{family}_base-all_loso-s{seed}.npz' for o in ORDERS}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('one or more output files already exist')
    scores = {o: np.full((n, 3), np.nan) for o in ORDERS}; fingerprints = {}
    folds = list(t7.protocol.folds(d, 'all_loso'))
    if len(folds) != 10:
        raise ValueError('expected ten held-out people')
    for fold, tr, te in folds[:1] if smoke else folds:
        m = t5.make(family, d, seed)
        if smoke:
            m.epochs = 1
        t5.attach_teacher(family, m, d, fold, tr, te, seed)
        m.fit(np.flatnonzero(tr))
        held = set(np.unique(d['subject'][te]).tolist())
        if held & m.fit_subjects or len(m.fit_subjects) != 9:
            raise ValueError('training-person exclusion failed')
        t5.check_am(family, m, d, te, 'all_loso'); t5.check_dg(family, m, d, te, 'all_loso')
        if m.n_sbn <= 0:
            raise ValueError('missing SessionBN')
        t5.check_cfg(family, m.net); m.set_k(t5.ARCHS[family][1])
        fingerprints[str(fold)] = fingerprint(m)
        rows = np.flatnonzero(te); streams = d['stream'][rows]
        for order in ORDERS:
            for stream in dict.fromkeys(streams):
                srows = rows[streams == stream]
                r = np.array(ordered_rows(srows, ranks[order][srows]))
                X = torch.from_numpy(m._aligned(r, causal=True)).to(m.dev)
                p = t7.stream(m, X, np.full((len(r), 3), 1/3), seed)['base']
                scores[order][r] = p
        print(f'fit {family} seed={seed} held={fold} fingerprint={fingerprints[str(fold)]}', flush=True)
    index = np.flatnonzero(np.isfinite(scores['within0']).all(axis=1))
    if not smoke and not np.array_equal(index, np.arange(n)):
        raise ValueError('production output does not cover canonical rows')
    for order, path in paths.items():
        p = scores[order][index]
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
            raise ValueError(f'{order}: invalid probabilities')
        with path.open('xb') as f:
            np.savez(f, index=index, proba=p)
    side = out / f'{family}_s{seed}_fit.json'
    with side.open('x') as f:
        json.dump({'family': family, 'seed': seed, 'fold_fingerprints': fingerprints,
                   'orders': ORDERS, 'smoke': smoke}, f, sort_keys=True, indent=2)
    print(f'created {len(paths)} streams; n={len(index)}; no scores computed', flush=True)


def reference_smoke(root, family):
    """Retrain the same single fold with the existing one-order runner."""
    t7 = load_module(root); d = t7.protocol.load()
    t7.TAG = 'ttt18within0'; t7.PERM = {'all_loso': permutations(root, len(d['y']))['within0']}
    refdir = root / 'runs/diag/ttt/lane1_smoke_reference'
    refdir.mkdir(parents=True, exist_ok=True)
    reference = refdir / f'ttt18within0_{family}_base-all_loso-s0.npz'
    if reference.exists():
        raise FileExistsError(reference)
    t7.run(family, d, 'all_loso', 0, refdir, epochs=1)
    actual = root / f'runs/diag/ttt/lane1_smoke/ttt18within0_{family}_base-all_loso-s0.npz'
    with np.load(reference) as old, np.load(actual) as new:
        if not np.array_equal(old['index'], new['index']):
            raise ValueError('one-order parity indices differ')
        diff = np.abs(old['proba'] - new['proba']).max()
        labels = np.count_nonzero(old['proba'].argmax(1) != new['proba'].argmax(1))
        if diff > 1e-6 or labels:
            raise ValueError(f'one-order parity failed: maxdiff={diff}, labels={labels}')
    print(f'one-order producer parity passed: maxdiff={diff:.3g} labels={labels}', flush=True)


if __name__ == '__main__':
    root = Path(os.environ['NI26_WS']).resolve()
    if len(sys.argv) == 3 and sys.argv[1] == 'smoke':
        family = FAMILIES[int(sys.argv[2])]
        generate(root, family, 0, smoke=True)
        reference_smoke(root, family)
    elif len(sys.argv) == 2:
        generate(root, *mapping(int(sys.argv[1])))
    else:
        raise SystemExit('usage: multistream.py TASK | smoke FAMILY_INDEX')
