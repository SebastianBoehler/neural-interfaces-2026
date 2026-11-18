"""Run frozen A6 with and without its explicit session normalization."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from wang_inputs import load, orders

BASE = Path('/mnt/lustre/work/gehler/gfh527')
sys.path.insert(0, str(BASE / 'ni26_joint382_v1_20261118'))
from restore import prior


@torch.no_grad()
def main(person, smoke):
    _, _, source_rows, runtime = prior.source()
    prior.owners(source_rows)
    assert runtime.align is False
    raw, names = load(person, smoke)
    with np.load(BASE / 'neural-interfaces-2026/data/proteus_windows.npz') as z:
        assert z['ch_names'].tolist() == names
    result = []
    for visit, wave in enumerate(raw, 1):
        for seed, rows in orders(person, visit, smoke):
            x = torch.from_numpy(runtime._aligned(rows, True, src=wave))
            for arm, k in [('A6', 20.), ('pop', None)]:
                runtime.net.spatial_conv[1].k = k
                v = runtime.net(x)
                assert torch.isfinite(v).all()
                torch.testing.assert_close(v[:6], runtime.net(x[:6]), rtol=1e-5, atol=3e-4)
                result.append(dict(visit=visit, seed=seed, arm=arm, rows=rows.tolist(), labels=v.argmax(1).tolist()))
    name = 'smoke' if smoke else str(person)
    Path(f'a6_{name}.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main(int(sys.argv[1]), len(sys.argv)>2)
