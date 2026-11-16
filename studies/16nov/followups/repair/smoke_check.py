"""Verify all real-data §302 smoke identities and source guards without scoring."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(HERE.parent / 'system')]
spec = importlib.util.spec_from_file_location('repair_score', HERE / 'score.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
s, m, old_root = r.s, r.m, r.OLD
with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
    n = len(z['y|all_loso']); sessions = len(np.unique(z['sess|all_loso']))
old = s.collect(old_root, n, sessions)
r.install_repair_guard()
m.ARMS = {a: ('eegn', .5, 'none', 1.) for a in ('E05','TI','PI','ET2','EPR25')}
m.ARMS['ZR'] = ('eegn', 0., 'none', 1.)
m.install_guard(); s.RECIPES['ZR'] = ('eegn', 'none', True)
remote = s.remote_manifest(ROOT / 'runs/lane1_inputs/streams.sha256')
with np.load(old_root / 'runs/newmember/s281_v66p_control_within0_gate.npz') as z: control = dict(z)
with np.load(ROOT / 'runs/newmember/s281_v66p_E05_within0_gate.npz') as z: reference = dict(z)
for arm in ('ZR','TI','PI','ET2','EPR25'):
    provenance = json.loads((ROOT / f'runs/newmember/s281_v66p_{arm}_within0_provenance.json').read_text())
    for seed in range(6):
        name = f'ttt18within0_eegn_base-all_loso-s{seed}.npz'
        found = [v for k, v in provenance['inputs_sha256'].items() if Path(k).name == name]
        if len(found) != 1 or remote.get(name) != found[0]: raise ValueError(f'{arm}: stream lock differs')
    with np.load(ROOT / f'runs/newmember/s281_v66p_{arm}_within0_gate.npz') as z:
        m.verify(control, dict(z), zero=arm == 'ZR')
        if arm in ('TI','PI'):
            if any(not np.array_equal(reference[k], z[k]) for k in reference): raise ValueError(f'{arm}: E05 chain identity')
    for base in s.BASES:
        value = s.load_output(ROOT, arm, 'within0', base, n, {})
        refarm = 'control' if arm == 'ZR' else 'E05'
        if arm in ('ZR','TI','PI'):
            ref = old['control', 'within0', base] if refarm == 'control' else s.load_output(ROOT, 'E05', 'within0', base, n, {})
            s.verify_zero(ref, value)
print('OK: old controls; ZR chain/CP; TI/PI E05 chain/CP; ET2/EPR25 gates and CP', flush=True)
