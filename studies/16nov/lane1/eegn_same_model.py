"""Compare multi-order's within0 path to the old runner with identical weights."""
import importlib.util
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(os.environ['NI26_WS']).resolve()
SPEC = importlib.util.spec_from_file_location('multistream', ROOT / 'studies/16nov/lane1/multistream.py')
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def main(family):
    if family not in ('eegn', 'eegn_dist2'):
        raise ValueError('unsupported family')
    t7 = M.load_module(ROOT); t5 = t7.T5; d = t7.protocol.load(); n = len(d['y'])
    fold, tr, te = next(iter(t7.protocol.folds(d, 'all_loso')))
    model = t5.make(family, d, 0); model.epochs = 1
    t5.attach_teacher(family, model, d, fold, tr, te, 0); model.fit(np.flatnonzero(tr))
    t5.check_dg(family, model, d, te, 'all_loso'); model.set_k(t5.ARCHS[family][1])
    initial = M.fingerprint(model)
    ranks = M.permutations(ROOT, n)['within0']; rows = np.flatnonzero(te)
    streams = d['stream'][rows]; actual = np.empty((n, 3))
    for stream in dict.fromkeys(streams):
        srows = rows[streams == stream]
        r = np.array(M.ordered_rows(srows, ranks[srows]))
        X = torch.from_numpy(model._aligned(r, causal=True)).to(model.dev)
        actual[r] = t7.stream(model, X, np.full((len(r), 3), 1/3), 0)['base']
    if M.fingerprint(model) != initial:
        raise ValueError('model state changed during new streaming path')
    t7.TAG = 'ttt18within0'; t7.PERM = {'all_loso': ranks}
    # The diagnostic bypasses only the second fit, preserving the original
    # runner's reference inference, delivery order, and output generation.
    t5.make = lambda *_: model
    model.fit = lambda *_: None
    out = ROOT / 'runs/diag/ttt/lane1_same_model'
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'ttt18within0_{family}_base-all_loso-s0.npz'
    if path.exists():
        raise FileExistsError(path)
    t7.run(family, d, 'all_loso', 0, out, epochs=1)
    if M.fingerprint(model) != initial:
        raise ValueError('model state changed during historical streaming path')
    with np.load(path) as z:
        if not np.array_equal(z['index'], rows):
            raise ValueError('same-model index mismatch')
        diff = float(np.max(np.abs(z['proba'] - actual[rows])))
        labels = int(np.count_nonzero(z['proba'].argmax(1) != actual[rows].argmax(1)))
    if diff > 1e-6 or labels:
        raise ValueError(f'same-model parity failed: {diff=} {labels=}')
    print(f'same-model parity passed {family}: {diff=} {labels=}', flush=True)


if __name__ == '__main__':
    main(('eegn', 'eegn_dist2')[int(sys.argv[1])])
