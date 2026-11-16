"""Verify §296 smoke and old guarded matrix, without calculating metrics."""
import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gated_score', HERE / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, root, old = m.s, m.ROOT, m.OLD
with np.load(root / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
    n = len(z['y|all_loso']); sessions = len(np.unique(z['sess|all_loso']))
old_outputs = s.collect(old, n, sessions)
m.install_guard(); remote = s.remote_manifest(root / 'runs/lane1_inputs/streams.sha256')
for arm in ('zero', 'EG06', 'DG08'):
    out = m.checked(arm, 'within0', n, {}, remote)
    if arm == 'zero':
        for base in s.BASES:
            s.verify_zero(old_outputs['control', 'within0', base], out[base])
print('OK: old matrix, package, routing, smoke masks, gate, streams, six CP and zero identity', flush=True)
