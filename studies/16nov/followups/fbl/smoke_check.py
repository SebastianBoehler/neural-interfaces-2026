"""Check new zero/dose and complete old controls without scoring."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
path = ROOT / 'studies/16nov/followups/system/score.py'
spec = importlib.util.spec_from_file_location('joint_score', path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = m.s
with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
    n = len(z['y|all_loso']); sessions = len(np.unique(z['sess|all_loso']))
old = s.collect(m.OLD, n, sessions)
m.install_guard()
with np.load(m.OLD / 'runs/newmember/s281_v66p_control_within0_gate.npz') as z:
    control = dict(z)
for arm in ('zero', 'FBL2'):
    with np.load(ROOT / f'runs/newmember/s281_v66p_{arm}_within0_gate.npz') as z:
        m.verify(control, dict(z), zero=arm == 'zero')
    for base in s.BASES:
        values = s.load_output(ROOT, arm, 'within0', base, n, {})
        if arm == 'zero': s.verify_zero(old['control', 'within0', base], values)
print('OK: old complete matrix, new zero and FBL2 gate/CP smoke', flush=True)
