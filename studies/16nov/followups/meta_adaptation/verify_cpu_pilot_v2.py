"""Verify the completed source-only CPU pilot without scoring labels."""
import json
from pathlib import Path
import numpy as np

root = Path('/mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/cpu_smoke_v2')
with np.load(root/'runs/s307/crossfit/P01_P02.npz',allow_pickle=False) as z:
    meta = json.loads(str(z['metadata'].item()))
    if (meta['outer'],meta['source'],meta['fit_people'],meta['smoke'],meta['held_accuracy_computed']) != ('P01','P02',[f'P{i:02d}' for i in range(3,11)],True,False):
        raise ValueError('CPU pilot source exclusion mismatch')
    if meta['calls'] <= 0 or meta['gradmax'] > 1e-5 or meta['fit_and_stream_seconds'] <= 0:
        raise ValueError('CPU pilot anchor did not converge')
    if z['theta'].shape != (3,44240) or z['features|within0'].shape[1] != 44239:
        raise ValueError('CPU pilot feature dimensions differ')
    print('CPU pilot fit/stream seconds',meta['fit_and_stream_seconds'],flush=True)
log = (root/'logs/s307-cpu-smoke-3115208.out').read_text()
if 'CPU real source features pass:' not in log or 'changed=0; no scores' not in log:
    raise ValueError('CPU reference/prefix parity did not pass')
print('CPU pilot source-only proof passed',flush=True)
