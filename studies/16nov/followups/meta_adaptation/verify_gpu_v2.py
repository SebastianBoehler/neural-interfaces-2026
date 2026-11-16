"""Read a completed no-score §307 pilot and verify live mechanism evidence."""
import json
from pathlib import Path
import numpy as np

root = Path('/mnt/lustre/work/gehler/gfh527/ni26_s307_meta_20261116/gpu_smoke_v2')
with np.load(root/'runs/s307/crossfit/P01_P02.npz', allow_pickle=False) as z:
    m = json.loads(str(z['metadata'].item()))
    if (m['outer'],m['source'],m['fit_people'],m['smoke'],m['held_accuracy_computed']) != ('P01','P02',[f'P{i:02d}' for i in range(3,11)],True,False):
        raise ValueError('pilot cross-fit exclusion/provenance failed')
    if m['gradmax'] > 1e-5 or m['calls'] <= 0 or m['fit_and_stream_seconds'] <= 0:
        raise ValueError('pilot fit did not converge')
    if z['theta'].shape != (3,44240) or z['features|within0'].shape[1] != 44239:
        raise ValueError('pilot strong core dimensions differ')
    print('pilot fitting and causal stream seconds',m['fit_and_stream_seconds'], flush=True)
for kind in ('scalar','blocks'):
    with np.load(root/f'runs/s307/meta/P01_{kind}_s0_smoke.npz', allow_pickle=False) as z:
        m = json.loads(str(z['metadata'].item())); p = z['parameter']
        if (m['outer'],m['kind'],m['seed'],m['smoke'],m['held_accuracy_computed']) != ('P01',kind,0,True,False):
            raise ValueError('pilot meta identity differs')
        if m['episodes'] != 1 or m['meta_gradient_max'] <= 0 or m['seconds'] <= 0 or not np.isfinite(p).all():
            raise ValueError('pilot meta gradient failed')
        if p.shape != (() if kind == 'scalar' else (51,)) or np.array_equal(p,np.full(p.shape,np.arctanh(.5))):
            raise ValueError('pilot meta parameters failed to move')
        print(kind,'pilot seconds',m['seconds'],'meta gradient',m['meta_gradient_max'],flush=True)
print('source-only pilot validated; no accuracy read',flush=True)
