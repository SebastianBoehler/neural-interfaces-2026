"""Verify already assembled §314 streams before the score-only retry."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

root=Path(os.environ['NI26_WS']).resolve(strict=True);out=root/'runs/s314/streams'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
orders=('within0','within1','within2','within3','full0','full1','runs0','runs1')
names={f'ttt18{o}_{family}_base-all_loso-s{s}.npz' for o in orders for family in ('tf314','tfd314') for s in (0,1)}
lock=json.loads((out/'validation.json').read_text())
entries={name:h for h,name in (line.split() for line in (out/'streams.sha256').read_text().splitlines())}
if set(entries)!=names or lock['streams']!=entries or lock['source_sha256']!=sha(root/'s314-source.sha256') or lock['held_accuracy_computed'] is not False:
 raise ValueError('stream lock/source is incomplete or changed')
folds={f'{p}_s{s}.npz' for p in (f'P{i:02d}' for i in range(1,11)) for s in (0,1)}
if set(lock['folds'])!=folds:raise ValueError('fold manifest incomplete')
for name,h in lock['folds'].items():
 if sha(root/'runs/s314/folds'/name)!=h:raise ValueError(f'changed fold {name}')
for o in orders:
 for family,arm in (('tf314','TF'),('tfd314','TFD')):
  for seed in (0,1):
   name=f'ttt18{o}_{family}_base-all_loso-s{seed}.npz';path=out/name
   if sha(path)!=entries[name]:raise ValueError(f'changed stream {name}')
   with np.load(path,allow_pickle=False) as z:
    m=json.loads(str(z['metadata'].item()));p=z['proba']
    if (m['arm'],m['seed'],m['order'],m['source_sha256'],m['fold_sha256'])!=(arm,seed,o,lock['source_sha256'],lock['folds']):
     raise ValueError(f'wrong stream identity {name}')
    if not np.array_equal(z['index'],np.arange(2655)) or p.shape!=(2655,3) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-10):
     raise ValueError(f'invalid stream {name}')
print('32 existing streams and 20 source folds verified; no accuracy read',flush=True)
