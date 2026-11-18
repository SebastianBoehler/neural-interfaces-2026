"""Apply only session evidence changes to the native stock rebalance trajectory."""
import json
import sys

import numpy as np

from capture import HERE, ORIG, main as run_actor

TRACE = ORIG.parent.parent / 'ni26_stage400_v1_20261118'


def main(arm, order):
    assert arm in ('RB', 'D', 'L')
    paths = [TRACE / f'{a}_{order}.npz' for a in ('control', 'RB', arm)]
    with np.load(paths[0]) as stock, np.load(paths[1]) as native, np.load(paths[2]) as actor:
        log = {}

        def transform(base, stage, value):
            if stage != 'rebal':
                return value
            mixed = stock[f'{base}|mix']
            assert np.array_equal(value, stock[f'{base}|rebal'])
            p = native[f'{base}|mix']
            q = actor[f'{base}|mix']
            assert (p > 0).all() and (q > 0).all()
            use = mixed.argmax(1) == value.argmax(1)
            updated = value.copy()
            ratio = q[use] / p[use]
            updated[use] *= ratio
            updated[use] /= updated[use].sum(1, keepdims=True)
            if arm != 'RB':
                assert np.max(abs(updated - value)) > 1e-4
            else:
                assert np.array_equal(ratio, np.ones_like(ratio))
            assert np.isfinite(updated).all() and np.allclose(updated.sum(1), 1)
            log[base] = dict(active=int(use.sum()), decisions=int((updated.argmax(1) != value.argmax(1)).sum()))
            return updated

        run_actor('control', order, transform, HERE / f't{arm}_{order}.npz')
    (HERE / f't{arm}_{order}.json').write_text(json.dumps(log))


if __name__ == '__main__':
    main(*sys.argv[1:])
