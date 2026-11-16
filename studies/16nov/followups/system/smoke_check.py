"""Verify complete smoke artifacts without calculating or reading treatment metrics."""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('joint_score', HERE / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, root, old = m.s, m.ROOT, m.OLD

with np.load(root / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
    n = len(z['y|all_loso']); sessions = len(np.unique(z['sess|all_loso']))
# Verify every old matched single and control, including route and all zero comparisons.
old_outputs = s.collect(old, n, sessions)
m.install_guard()
lock = s.remote_manifest(root / 'runs/lane1_inputs/streams.sha256')
with np.load(old / 'runs/newmember/s281_v66p_control_within0_gate.npz') as z:
    control = dict(z)
for arm in ('zero', 'E05CS', 'FBL15'):
    with np.load(root / f'runs/newmember/s281_v66p_{arm}_within0_gate.npz') as z:
        m.verify(control, dict(z), zero=arm == 'zero')
    p = json.loads((root / f'runs/newmember/s281_v66p_{arm}_within0_provenance.json').read_text())
    add, swap, _ = m.expected(arm)
    for family in [f for group in (add, swap) if group != 'none' for f in group.split(',')]:
        for seed in range(6):
            name = f'ttt18within0_{family}_base-all_loso-s{seed}.npz'
            actual = [v for k, v in p['inputs_sha256'].items() if Path(k).name == name]
            if len(actual) != 1 or actual[0] != lock.get(name):
                raise ValueError(f'{arm}: stream hash {name}')
    for base in s.BASES:
        values = s.load_output(root, arm, 'within0', base, n, {})
        if arm == 'zero':
            s.verify_zero(old_outputs['control', 'within0', base], values)
print('OK: old complete matrix, new smoke gates, stream locks, six CP variants and zero equality', flush=True)
