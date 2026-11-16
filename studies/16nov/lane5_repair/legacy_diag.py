"""Compare stored §293 P01 streams with one-device checkpoint replay, no labels."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path.insert(0, str(ROOT / 'studies/16nov/lane5'))
from bridge import Native, Session  # noqa: E402
from run import ARMS, probability, sha  # noqa: E402


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is unavailable')
    with np.load(ROOT / 'data/proteus_windows.npz', allow_pickle=False) as z:
        x, subj, sess = (z[k].copy() for k in ('X', 'subject', 'session'))
    with np.load(ROOT / 'runs/shuffle_perms.npz', allow_pickle=False) as z:
        perm = z['within|0|all_loso'].astype(np.int64)
    rows = perm[np.char.endswith(subj[perm], '/P01')]
    stream = np.char.add(np.char.add(subj, '|'), sess)
    native = Native(os.environ['LANE5_LIB']); report = {}
    for arm, arch, _ in ARMS:
        cp = ROOT / f'runs/lane5_checkpoints/{arm}_s0_P01.pt'
        ref = ROOT / f'runs/lane5/{arm}_s0_P01.npz'
        model = native.create(arch, 1, 0); model.load(cp)
        with np.load(ref, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            if sha(cp) != meta['checkpoint_sha256'] or not np.array_equal(z['within0|index'], rows):
                raise ValueError('checkpoint or order mismatch')
            if hashlib.sha256(model.params().tobytes()).hexdigest() != meta['parameter_sha256']:
                raise ValueError('checkpoint parameter hash mismatch')
            pred = {k: np.empty((len(rows), 3)) for k in ('static', 'adaptive')}
            for session in dict.fromkeys(stream[rows]):
                loc = np.flatnonzero(stream[rows] == session); a, b = Session(model), Session(model)
                for i in loc:
                    pred['static'][i] = probability(a.step(x[rows[i]], False))
                    pred['adaptive'][i] = probability(b.step(x[rows[i]], True))
                a.solver.close(); b.solver.close()
            stats = {}
            for key in pred:
                target = z[f'within0|{key}']; delta = np.abs(pred[key] - target)
                margin = np.sort(target, axis=1)[:, -1] - np.sort(target, axis=1)[:, -2]
                safe = margin > 4e-5
                stats[key] = {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
                              'rows_over_2e-5': int((delta.max(1) > 2e-5).sum()),
                              'nontied_label_changes': int((pred[key][safe].argmax(1) != target[safe].argmax(1)).sum()),
                              'near_ties': int((~safe).sum())}
            report[arm] = {'checkpoint_sha': sha(cp), 'original_gpu': meta['gpu_name'], 'checks': stats}
        model.close()
    print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'arms': report}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
