"""Paired §299 dose contrasts using the unchanged complete-matrix scorer."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SYSTEM = HERE.parent / 'system'
sys.path.insert(0, str(SYSTEM))
from rolling_audit import paths_for, record
spec = importlib.util.spec_from_file_location('joint_score', SYSTEM / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, OLD = m.s, m.OLD


def main():
    prereg = ROOT / 'runs/lane1_inputs/fbl-registration.md'
    if not prereg.is_file() or '### 299.' not in prereg.read_text(): raise ValueError('§299 registration required')
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, len(y), sessions); new = m.collect_new(len(y), {})
    for order in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(ROOT / f'runs/newmember/s281_v66p_zero_{order}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for b in s.BASES: s.verify_zero(old['control', order, b], s.load_output(ROOT, 'zero', order, b, len(y), {}))
    def person(arm, order):
        return np.mean([s.person_accuracy(new[arm, order, b], y, cells, people, s.PEOPLE) for b in s.BASES], axis=0)
    counts = np.array([len(np.unique(cells[people == p])) for p in s.PEOPLE])
    paths = paths_for(ROOT, ['FBL05','FBL15','FBL2','FBL3','zero'], s.ORDERS, s.BASES)
    paths += paths_for(OLD, ['control','zero_eegn'], s.ORDERS, s.BASES)
    audit, sha, previous = record('299', 'paired_doses', paths, prereg, s.sha256)
    print('Sequential dev dose comparisons; repeated selection; no package or returning claim.')
    print('audit', audit, 'entry_sha256', sha, 'already_read_blocks', previous)
    for arm in ('FBL05','FBL2','FBL3'):
        d = np.stack([person(arm, order) - person('FBL15', order) for order in s.ORDERS])
        print(arm, '- FBL15', s.summarize(d, s.ORDERS, s.PEOPLE), 'official_equal_cell',
              s.cell_weighted_summary(d, s.ORDERS, counts), flush=True)


if __name__ == '__main__': main()
