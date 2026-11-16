"""Unlabelled, session-local raw/CAR covariance spectra; never load command labels."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
CUTS = (1e-3, 1e-4, 1e-6)
DATA_SHA = '2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327'
PERM_SHA = '6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''): h.update(x)
    return h.hexdigest()


def spectra(cov):
    cov = (cov + cov.T) * .5
    w = np.linalg.eigvalsh(cov); scale = np.trace(cov) / 41
    if not np.isfinite(w).all() or not np.isfinite(scale) or scale <= 0 or w[0] < -1e-10 * scale:
        raise ValueError('invalid covariance spectrum')
    positive = w[w > scale * 1e-12]
    return dict(eigenvalues_over_trace41=(w / scale).tolist(),
                below={str(c): int(np.sum(w < c * scale)) for c in CUTS},
                positive_condition=float(w[-1] / positive[0]), trace41=float(scale))


def window_covariances(x):
    if x.ndim != 3 or x.shape[1:] != (41, 480) or not np.isfinite(x).all():
        raise ValueError('wrong input tensor')
    out = np.empty((len(x), 41, 41), dtype=np.float64)
    for start in range(0, len(x), 32):
        block = x[start:start + 32].astype(np.float64)
        block -= block.mean(axis=-1, keepdims=True)
        out[start:start + len(block)] = block @ block.transpose(0, 2, 1) / 480
    return out


def record(cov, projection):
    return {'raw': spectra(cov), 'car': spectra(projection @ cov @ projection)}


def main():
    root = Path(os.environ['AUDIT_ROOT']).resolve(strict=True)
    data = Path(os.environ['AUDIT_DATA']).resolve(strict=True)
    permutations = Path(os.environ['AUDIT_PERM']).resolve(strict=True)
    output = root / 'runs/rank306/results.json'
    if output.exists(): raise FileExistsError(output)
    if sha(data) != DATA_SHA or sha(permutations) != PERM_SHA:
        raise ValueError('input digest mismatch')
    with np.load(data, allow_pickle=False) as z:
        # Deliberately do not access y, task, onset or run.
        x, subjects, sessions, interface = (z[k].copy() for k in ('X', 'subject', 'session', 'interface'))
    if len(x) != 2655 or set(interface) != {'BrainHero', 'Graz'}:
        raise ValueError('unexpected data inventory')
    cov = window_covariances(x); del x
    projector = np.eye(41) - np.ones((41, 41)) / 41
    with np.load(permutations, allow_pickle=False) as z:
        perms = {o: z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
    if any(not np.array_equal(np.sort(p), np.arange(len(cov))) for p in perms.values()):
        raise ValueError('invalid pinned order')
    groups = sorted(set(zip(subjects, sessions))); rows = []
    for person, session in groups:
        mask = (subjects == person) & (sessions == session)
        for order, perm in perms.items():
            ix = perm[mask[perm]]; sequence = np.cumsum(cov[ix], axis=0)
            for n in sorted(set((1, min(10, len(ix)), min(40, len(ix)), len(ix)))):
                rows.append(dict(person=person, session=session, order=order, prefix=n,
                                 kind='prefix', **record(sequence[n - 1] / n, projector)))
            left, right = (ix[interface[ix] == v] for v in ('BrainHero', 'Graz'))
            n = min(len(left), len(right))
            if not n: raise ValueError('session has no matched-interface samples')
            for label, group in (('BrainHero', left), ('Graz', right)):
                rows.append(dict(person=person, session=session, order=order, prefix=n,
                                 kind='interface_descriptive', interface=label,
                                 **record(cov[group[:n]].mean(axis=0), projector)))
    if sha(data) != DATA_SHA or sha(permutations) != PERM_SHA:
        raise ValueError('input changed during audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(registration='§306', data_sha256=DATA_SHA, permutation_sha256=PERM_SHA,
                  code_sha256=sha(__file__), session_count=len(groups), rows=rows,
                  command_labels_opened=False, interpretation='descriptive only; no model selection')
    with output.open('x') as f: json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    print('saved label-blind rank audit', output, 'sha256', sha(output), flush=True)


if __name__ == '__main__': main()
