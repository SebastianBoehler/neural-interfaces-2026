"""Require exact archived-control equality after outer chain and CP replay."""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
BASES = ('B2_pool', 'B2_s345', 'B2_s012', 'B2_s0', 'B2_s1', 'B2_s2')


def compare_files(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError(f'different chain keys: {first}, {second}')
        for key in a.files:
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f'zero chain mismatch: {key}')


def main(order):
    compare_files(ROOT / f'runs/chainout/s281_v66p_control_{order}.npz',
                  ROOT / f'runs/chainout/s281_v66p_geo_zero_{order}.npz')
    compare_files(ROOT / f'runs/newmember/s281_v66p_control_{order}_gate.npz',
                  ROOT / f'runs/newmember/s281_v66p_geo_zero_{order}_gate.npz')
    for base in BASES:
        ctl = ROOT / f'runs/s281_v66_cp/control_{order}_{base}.npz'
        zero = ROOT / f'runs/s281_v66_cp/geo_zero_{order}_{base}.npz'
        with np.load(ctl, allow_pickle=False) as a, np.load(zero, allow_pickle=False) as b:
            if not np.array_equal(a['scores'], b['scores']):
                raise ValueError(f'zero personalisation mismatch: {order}/{base}')
    print(f'EXACT ZERO: complete chain, gate and six CP bases, {order}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: verify_zero.py ORDER')
    main(sys.argv[1])
