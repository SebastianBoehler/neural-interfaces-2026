"""Validate ENU and fresh zero without reading outcome metrics."""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('neutral_score', HERE / 'score.py')
x = importlib.util.module_from_spec(spec); spec.loader.exec_module(x)
s, m, ROOT, OLD = x.s, x.m, x.ROOT, x.OLD


def main():
    y, _, _, old = x.setup()
    m.ROOT = ROOT; m.ARMS = {'ENU': ('eegn', .5, 'none', 1.)}; m.install_guard()
    remote = s.remote_manifest(ROOT / 'runs/lane1_inputs/streams.sha256')
    provenance = json.loads((ROOT / 'runs/newmember/s281_v66p_ENU_within0_provenance.json').read_text())
    for seed in range(6):
        name = f'ttt18within0_eegn_base-all_loso-s{seed}.npz'
        found = [v for k,v in provenance['inputs_sha256'].items() if Path(k).name == name]
        if found != [remote[name]]: raise ValueError('ENU source stream mismatch')
    with np.load(OLD/'runs/newmember/s281_v66p_control_within0_gate.npz') as a, np.load(ROOT/'runs/newmember/s281_v66p_ENU_within0_gate.npz') as b:
        m.verify(dict(a),dict(b),zero=False)
    for base in s.BASES:
        s.load_output(ROOT,'ENU','within0',base,len(y),{})
        side = json.loads((ROOT/f'runs/s281_v66_cp/ENU_within0_{base}.json').read_text())
        if not any(k.endswith('/followups/neutral/cp.py') for k in side['inputs_sha256']):
            raise ValueError('neutral CP source not tracked')
    x.zero_guards(old,len(y))
    m.ARMS['E05'] = ('eegn', .5, 'none', 1.)
    s.RECIPES['E05'] = ('eegn', 'none', False)
    for arm in ('TI','PI'):
        for base in s.BASES:
            a = s.load_output(ROOT,arm,'within0',base,len(y),{})
            b = s.load_output(ROOT,'E05','within0',base,len(y),{})
            s.verify_zero(a,b)
    print('ENU guarded smoke: streams, gate, six CP, eight zeros and two identity controls pass; no metrics read.')


if __name__ == '__main__': main()
