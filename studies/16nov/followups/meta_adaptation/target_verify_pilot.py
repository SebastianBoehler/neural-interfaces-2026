"""Check the real P01 target smoke before releasing other target people."""
import json
from pathlib import Path
import numpy as np
root=Path.cwd(); held='P01'; path=root/'runs/s307/target/P01.npz'
with np.load(path,allow_pickle=False) as z:
    m=json.loads(str(z['metadata'].item()))
    if m['held'] != held or m['held_accuracy_computed'] is not False or m['fit_gradmax'] > 1e-5 or m['max_delta_ratio'] > .03000001:
        raise ValueError('P01 target guard metadata failed')
    if set(m['updates']) != {'within0','within1','within2','within3','full0','full1','runs0','runs1'} or min(m['updates'].values()) <= 0:
        raise ValueError('P01 target update coverage missing')
    print('P01 complete target seconds',m['seconds'],'update counts',m['updates'],flush=True)
print('P01 actual target streams pass no-score metadata guard',flush=True)
