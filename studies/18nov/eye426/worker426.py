"""Run each counterfactual through an independent exact-archive session."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import worker as frozen


def main(arm, p, smoke):
    assert 'A100' in torch.cuda.get_device_name() and torch.__version__ == '2.14.1+cu130'
    prior = json.loads(Path(f'/prior/{arm}_{6+(p-1)//5}.json').read_text())
    ref = {r['seed']: r for r in prior if r['kind'] == 'W' and r['person'] == p}
    assert set(ref) == {415, 1415}
    output = []
    with np.load(f'person{p}.npz') as z:
        model = frozen.replay.load(Path('/package'), z['names'].tolist())
        for mode in ('original', 'aligned', 'reversed'):
            x = torch.from_numpy(z[mode])
            assert x.shape == (60, 41, 480) and x.dtype == torch.float32
            for seed in ((415,) if smoke else (415, 1415)):
                rows = np.random.default_rng([415,p,seed,0]).permutation(60)
                assert np.array_equal(z['ids'][rows], ref[seed]['raw'])
                rec = frozen.run(model, x, rows, arm)
                assert rec['rows'] == ref[seed]['rows']
                if mode == 'original':
                    assert rec['labels'] == ref[seed]['labels']
                if smoke:
                    assert frozen.run(model, x, rows[:7], arm)['labels'] == rec['labels'][:7]
                output.append(dict(person=p, seed=seed, mode=mode, arm=arm,
                                   raw=z['ids'][rows].tolist(), **rec))
    tag = 'smoke' if smoke else 'score'
    Path(f'{tag}-{arm}-{p}.json').write_text(json.dumps(output))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), '--smoke' in sys.argv)
