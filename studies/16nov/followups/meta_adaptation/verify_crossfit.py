"""Verify and pin all 90 complete source-only crossfit artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
people = [f'P{i:02d}' for i in range(1,11)]
cache = {Path(n).stem: h for h,n in (line.split() for line in (ROOT/'cache.sha256').read_text().splitlines())}
paths = []
for outer in people:
    for source in people:
        if source == outer: continue
        path = ROOT/f'runs/s307/crossfit/{outer}_{source}.npz'
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            fit = [p for p in people if p not in (outer,source)]
            if (meta['outer'],meta['source'],meta['fit_people'],meta['smoke'],meta['held_accuracy_computed']) != (outer,source,fit,False,False):
                raise ValueError(f'{path}: source exclusion/provenance mismatch')
            if meta['input_sha256'] != {p: cache[p] for p in people if p != outer}:
                raise ValueError(f'{path}: cached source hashes differ')
            if meta['gradmax'] > 1e-5 or meta['calls'] <= 0 or z['theta'].shape != (3,44240):
                raise ValueError(f'{path}: anchor solver failed')
            for name in ('within0','full0','runs0'):
                if z[f'features|{name}'].shape != (len(z['source_rows']),44239):
                    raise ValueError(f'{path}: causal source stream incomplete')
        paths.append(path)
if len(paths) != 90: raise ValueError('90 crossfits required')
with (ROOT/'crossfit.sha256').open('x') as out:
    for path in paths:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''): h.update(b)
        print(h.hexdigest(), path.relative_to(ROOT), file=out)
print('all 90 source-only crossfits verified; no score read', flush=True)
