"""Read complete preregistered matched interactions with the existing guards."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rolling_audit import paths_for, record
spec = importlib.util.spec_from_file_location('joint_score', HERE / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, root, old = m.s, m.ROOT, m.OLD


def main():
    prereg = root / 'runs/lane1_inputs/rolling-registration.md'
    if not prereg.is_file() or '### 298.' not in prereg.read_text():
        raise ValueError('§298 registration required')
    with np.load(root / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old_results = s.collect(old, len(y), sessions)
    m.ARMS = {a: m.ARMS[a] for a in ('E05CS', 'E05', 'EFBL', 'FBL15')}
    current = m.collect_new(len(y), {})
    for order in s.ORDERS:
        with np.load(old / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(root / f'runs/newmember/s281_v66p_zero_{order}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for base in s.BASES:
            s.verify_zero(old_results['control', order, base], s.load_output(root, 'zero', order, base, len(y), {}))
    def metric(output, arm, order):
        return np.mean([s.person_accuracy(output[arm, order, b], y, cells, people, s.PEOPLE) for b in s.BASES], axis=0)
    counts = np.array([len(np.unique(cells[people == p])) for p in s.PEOPLE])
    for arm, single, external in (('E05CS', 'E05', 'C10S'), ('EFBL', 'FBL15', 'E')):
        delta = np.stack([metric(current, arm, o) - metric(current, single, o)
                          - metric(old_results, external, o) + metric(old_results, 'control', o) for o in s.ORDERS])
        paths = paths_for(root, [arm, single, 'zero'], s.ORDERS, s.BASES)
        paths += paths_for(old, [external, 'control', 'zero_eegn'], s.ORDERS, s.BASES)
        audit, sha, previous = record('294', f'interaction:{arm}', paths, prereg, s.sha256)
        print('Sequential dev interaction; repeated selection, full wave still required.')
        print('audit', audit, 'entry_sha256', sha, 'already_read_blocks', previous)
        print(arm, '-', single, '-', external, '+ control',
              s.summarize(delta, s.ORDERS, s.PEOPLE), 'official_equal_cell',
              s.cell_weighted_summary(delta, s.ORDERS, counts), flush=True)


if __name__ == '__main__': main()
