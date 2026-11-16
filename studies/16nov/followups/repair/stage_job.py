"""Guard the complete §302 matrix, then report its saved stage locations."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(HERE.parent / 'system')]
spec = importlib.util.spec_from_file_location('repair_score', HERE / 'score.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
s, m, OLD = r.s, r.m, r.OLD
from stage import report_stages


def main():
    prereg = ROOT / 'runs/lane1_inputs/repair-registration.md'
    if not prereg.is_file() or '### 302.' not in prereg.read_text(): raise ValueError('registration missing')
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, len(y), sessions)
    r.install_repair_guard()
    m.ARMS = {a: ('eegn', .5, 'none', 1.) for a in ('E05', 'ET05', 'ET2', 'EPR25', 'EPR50')}
    new = m.collect_new(len(y), {})
    m.ARMS['ZR'] = ('eegn', 0., 'none', 1.); s.RECIPES['ZR'] = ('eegn', 'none', True)
    for order in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(ROOT / f'runs/newmember/s281_v66p_ZR_{order}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for base in s.BASES:
            s.verify_zero(old['control', order, base], s.load_output(ROOT, 'ZR', order, base, len(y), {}))
    report_stages(ROOT, OLD, old, new, y, cells, people, s, prereg)


if __name__ == '__main__': main()
