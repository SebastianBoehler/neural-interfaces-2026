"""Check the historical native bank and reapply the same policy to §430 stages."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from stocks456 import NATIVE, STAGES, module, ref


def build(meta, ages, stock, smoke):
    policy = module(NATIVE, 'policy', '59e3b26ffa371f9cb9e12663f49214063e54b5f67fb6a232ec6c30c780682df2')
    owner = NATIVE / 'run.py'
    assert ref.sha(owner) == 'e7a1aa1fb573d10b9805a16c3a39dca960bf940277665d2b72f0efd6c43c75e6'
    from fusion_inputs import OWNER
    env = {**os.environ, 'PYTHONPATH':os.environ['PYTHONPATH']+os.pathsep+str(OWNER)}
    out = {}
    for order in ref.ORDERS[:1] if smoke else ref.ORDERS:
        name = order+'.npz'
        source = ref.source_manifest(NATIVE/'source.sha256')
        proof = json.loads((NATIVE/(order+'.json')).read_text())
        assert proof['source'] == source and proof['bank'] == ref.sha(NATIVE/name)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work/'source.sha256').symlink_to(NATIVE/'source.sha256')
            subprocess.run([sys.executable,str(owner),order], cwd=work, env=env, check=True)
            with np.load(work/name) as rerun, np.load(NATIVE/name) as measured:
                assert set(rerun.files) == set(measured.files)
                assert all(np.array_equal(rerun[k],measured[k]) for k in rerun)
        path = STAGES / f'stages_{order}.npz'
        proof = json.loads(path.with_suffix('.json').read_text())
        assert proof['order'] == order and ref.sha(path) == proof['output']
        assert ref.source_manifest(STAGES/'source.sha256') == proof['experiment']
        sessions = []
        for s in np.unique(meta['stream']):
            ids = np.flatnonzero(meta['stream'] == s)
            rows = ids[np.argsort(ages[order][ids])]
            assert np.array_equal(ages[order][rows], np.arange(len(rows)))
            sessions.append(rows)
        with np.load(path) as z:
            assert np.array_equal(z['terminal'], stock[order])
            shifts, _ = policy.corrections(z['raw'], z['filtered'], z['terminal'], sessions)
            shifts['stock'] = np.zeros_like(z['terminal'])
            for arm,v in shifts.items():
                labels = (z['terminal']+v).argmax(1).astype(np.uint8)
                assert np.array_equal(labels[ages[order]<90],stock[order][ages[order]<90].argmax(1))
                for person in ref.PEOPLE:
                    out[person,order,arm] = labels[meta['person']==person]
    return out
