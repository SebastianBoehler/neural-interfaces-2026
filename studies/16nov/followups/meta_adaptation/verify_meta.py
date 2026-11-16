"""Pin the complete CPU-trained source meta-model matrix before target streaming."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

lock = sha(ROOT/'meta-cpu-src.sha256')
people = [f'P{i:02d}' for i in range(1,11)]
paths = []
for outer in people:
    for kind in ('scalar','blocks'):
        for seed in (0,1):
            path = ROOT/f'runs/s307/meta/{outer}_{kind}_s{seed}.npz'
            route = path.with_suffix('.cpu-route').read_text().split()
            if route != ['CPU',sha(path),lock]: raise ValueError(f'{path}: CPU route changed')
            with np.load(path,allow_pickle=False) as z:
                m=json.loads(str(z['metadata'].item())); p=z['parameter']
                expected=[v for v in people if v != outer]
                if (m['outer'],m['kind'],m['seed'],m['smoke'],m['held_accuracy_computed'],m['source_people']) != (outer,kind,seed,False,False,expected):
                    raise ValueError(f'{path}: source metadata mismatch')
                if m['episodes'] <= 0 or m['meta_gradient_max'] <= 0 or not np.isfinite(p).all() or p.shape != (() if kind=='scalar' else (51,)):
                    raise ValueError(f'{path}: inactive or nonfinite meta model')
            paths.append(path)
if len(paths) != 40: raise ValueError('expected 40 meta models')
with (ROOT/'meta.sha256').open('x') as f:
    for path in paths: print(sha(path),path.relative_to(ROOT),file=f)
print('40 source-only CPU meta models verified and pinned; no target scores',flush=True)
