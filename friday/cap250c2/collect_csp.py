"""Score full cap250 readouts after each held decision bank is locked."""
import json
from pathlib import Path

import numpy as np
import score379 as ref
from collect import interval
from measure import official
from produce import public_inputs

from run_csp import ARMS


def score():
    source = ref.sha(Path('source.sha256'))
    banks, receipts = {}, {}
    for order in ref.ORDERS:
        path = Path(order+'.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert receipt['order'] == order and receipt['source'] == source
        assert ref.sha(path) == receipt['output']
        with np.load(path) as z:
            assert set(z.files) == set(ARMS)
            banks[order] = dict(z)
        receipts[order] = receipt
    Path('before_labels.json').write_text(json.dumps(receipts, indent=2))

    meta, _ = public_inputs()
    with np.load(ref.BASE/'data/proteus_windows.npz') as z:
        labels = z['y']
    metric = official()
    tables = {a: np.zeros((10, 8)) for a in ARMS}
    weights = np.zeros(10)
    for i, person in enumerate(ref.PEOPLE):
        mask = meta['person'] == person
        for j, order in enumerate(ref.ORDERS):
            for name in ARMS:
                s = metric(labels[mask], banks[order][name][mask], meta, mask)
                weights[i] = s['n_cells']
                tables[name][i, j] = 100*s['balanced_accuracy_cell_mean']
    assert weights.sum() == 28
    absolute = {a: interval(v, weights) for a, v in tables.items()}
    assert abs(absolute['stock']['pp']-76.103915)<1e-5
    pairs = (('shape_fit', 'stock'), ('shape_fit', 'shape_all'), ('shape_all', 'stock'))
    contrasts = {}
    for a, b in pairs:
        diff = interval(tables[a]-tables[b], weights)
        counts = dict(changed=0, corrections=0, regressions=0)
        for bank in banks.values():
            right_a, right_b = bank[a] == labels, bank[b] == labels
            counts['changed'] += int(np.count_nonzero(bank[a] != bank[b]))
            counts['corrections'] += int(np.count_nonzero(right_a & ~right_b))
            counts['regressions'] += int(np.count_nonzero(right_b & ~right_a))
        contrasts[a+'-'+b] = {**diff, **counts}
    Path('result.json').write_text(json.dumps(dict(people=ref.PEOPLE, absolute=absolute,
        contrasts=contrasts, joint_pp_primary=.566*contrasts['shape_fit-stock']['pp'],
        receipt=ref.sha(Path('before_labels.json'))), indent=2))


if __name__ == '__main__':
    score()
