"""Load the pinned §454 H100 predictions and calculate their registered corrections."""
import json
from pathlib import Path

import numpy as np
import score379 as ref


def load(root, kind, seed, person, rows):
    root = Path(root)
    stem = root / f'{kind}_{seed}_{person}.npz'
    record = stem.with_suffix('.json')
    actor = json.loads(record.read_text())
    assert actor['source'] == ref.sha(root/'actor_source.sha256') and actor['output'] == ref.sha(stem)
    assert actor['kind'] == kind and actor['held'] == person and actor['seed'] == seed
    assert actor['orders'] == ref.ORDER_SHA and actor['data'] == ref.DATA_SHA
    assert set(actor['train_people']) == set(ref.PEOPLE) - {person}
    with np.load(stem) as z:
        assert np.array_equal(rows, z['index'])
        bank = {a:z[a].astype(np.float64) for a in ('memory','donor','source')}
    assert all(v.shape == (8,len(rows),3) and np.isfinite(v).all() for v in bank.values())
    deltas = {a:v-bank['source'] for a,v in bank.items()}
    corrected = {a:.1*np.clip(d-d.mean(-1,keepdims=True),-1,1) for a,d in deltas.items()}
    assert np.all(corrected['source'] == 0)
    return corrected, ref.sha(record)
