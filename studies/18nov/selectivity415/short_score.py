"""Compare package effects after full and short history on identical rows."""
import json
from pathlib import Path

import numpy as np


def decisions(folder, arm, shards):
    rows = {}
    for i in shards:
        for r in json.loads((folder / f'{arm}_{i}.json').read_text()):
            k = r['person'], r['seed']
            assert k not in rows
            rows[k] = r
    return rows


def main():
    banks = [decisions(path, arm, shards) for path, shards in
             [(Path('../ni26_select415_20261118'), range(6, 12)), (Path('.'), range(6))]
             for arm in ('old', 'cap')]
    assert len(banks[0]) == 52 and all(b.keys() == banks[0].keys() for b in banks)
    effects = []
    with np.load('../neural-interfaces-2026/data/shin2018/windows.npz') as z:
        y, run = z['y'], z['run']
    for person in range(1, 27):
        per_order = []
        for seed in (415, 1415):
            r = [b[person, seed] for b in banks]
            raw = r[0]['raw']
            assert all(o['raw'] == raw and o['groups'] == r[0]['groups'] for o in r)
            pred = np.array([o['labels'] for o in r])
            assert np.array_equal(pred[:2, :20], pred[2:, :20])
            class_contrast = (pred == 2).astype(int) - (pred == 1).astype(int)
            cells = []
            for g in set(run[raw]):
                mask = run[raw] == g
                on, off = mask & (y[raw] == 1), mask & (y[raw] == 0)
                assert on.sum() == off.sum() == 10
                cells.append(100 * (class_contrast[:, on].mean(1) - class_contrast[:, off].mean(1)))
            per_order.append(np.mean(cells, axis=0))
        effects.append(np.mean(per_order, axis=0))
    full_old, full_cap, short_old, short_cap = np.array(effects).T
    full, short = full_cap - full_old, short_cap - short_old
    delta = full - short
    primary = json.loads(Path('../ni26_select415_20261118/result.json').read_text())
    np.testing.assert_allclose(full, primary['W']['per_person'], atol=1e-10, rtol=0)
    ci = np.quantile(np.random.default_rng(415).choice(delta, (10000, 26)).mean(1), [.1, .9])
    Path('result.json').write_text(json.dumps(dict(full=float(full.mean()), short=float(short.mean()),
        difference=float(delta.mean()), ci80=ci.tolist(), person=delta.tolist(), prefix_decisions_matched=2080), indent=2)+'\n')


if __name__ == '__main__':
    main()
