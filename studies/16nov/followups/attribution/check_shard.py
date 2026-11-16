"""Read-only guard for the replacement §300 control/dose smoke shard."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(HERE.parent / 'system')]
import metrics
import shard_guard
spec = importlib.util.spec_from_file_location('joint_score', HERE.parent / 'system/score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
arrays = shard_guard.load(ROOT, 'within0', ('control','E05CS'), ('B2_pool',), m.s, metrics.probability, smoke=True)
if len({x.shape for x in arrays.values()}) != 1: raise ValueError('smoke array shapes differ')
for arm, root in (('control', m.OLD), ('E05CS', ROOT)):
    with np.load(root / f'runs/chainout/s281_v66p_{arm}_within0.npz') as z:
        if not np.array_equal(z['B2_pool|V6Fr'], arrays[f'{arm}|V6Fr|B2_pool']): raise ValueError('smoke raw mismatch')
print('OK: replacement smoke source hashes, complete guards and six stage arrays', flush=True)
