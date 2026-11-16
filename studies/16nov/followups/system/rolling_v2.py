"""Guard and score one complete §294 arm using the existing scorer primitives."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rolling_audit import paths_for, record

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('joint_score', HERE / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, root, old = m.s, m.ROOT, m.OLD


def prerequisites(arm):
    if arm not in m.ARMS:
        raise ValueError('unknown registered §294 arm')
    # No partial orders or base variants are accepted.
    paths = [root / f'runs/s281_v66_cp/{arm}_{order}_{base}.npz' for order in s.ORDERS for base in s.BASES]
    if not all(p.is_file() and p.with_suffix('.json').is_file() for p in paths):
        raise ValueError(f'{arm}: complete eight-order, six-base output required')


def main():
    if len(sys.argv) != 2:
        raise ValueError('supply exactly one complete registered arm')
    arm = sys.argv[1]
    prereg = root / 'runs/lane1_inputs/rolling-registration.md'
    if not prereg.is_file() or '### 298.' not in prereg.read_text():
        raise ValueError('rolling §298 registration required before scoring')
    prerequisites(arm)
    with np.load(root / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old_results = s.collect(old, len(y), sessions)
    m.ARMS = {arm: m.ARMS[arm]}
    current = m.collect_new(len(y), {})
    for order in s.ORDERS:
        with np.load(old / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(root / f'runs/newmember/s281_v66p_zero_{order}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for base in s.BASES:
            s.verify_zero(old_results['control', order, base], s.load_output(root, 'zero', order, base, len(y), {}))
    def metric(output, o):
        return np.mean([s.person_accuracy(output[o, b], y, cells, people, s.PEOPLE) for b in s.BASES], axis=0)
    delta = []
    for order in s.ORDERS:
        this = {(order, b): current[arm, order, b] for b in s.BASES}
        control = {(order, b): old_results['control', order, b] for b in s.BASES}
        delta.append(metric(this, order) - metric(control, order))
    counts = np.array([len(np.unique(cells[people == p])) for p in s.PEOPLE])
    paths = paths_for(root, [arm, 'zero'], s.ORDERS, s.BASES) + paths_for(old, ['control', 'zero_eegn'], s.ORDERS, s.BASES)
    audit, entry_sha, previous = record('294', arm, paths, prereg, s.sha256)
    print('Sequential dev read; repeated selection; full-matrix report still required.')
    print('audit', audit, 'entry_sha256', entry_sha, 'already_read_blocks', previous, flush=True)
    print(f'{arm} - control', s.summarize(np.stack(delta), s.ORDERS, s.PEOPLE),
          'official_equal_cell', s.cell_weighted_summary(np.stack(delta), s.ORDERS, counts), flush=True)
    # Interactions require separate, complete, guarded matched-single reads.
    print('No interaction inferred by this single-arm read.', flush=True)


if __name__ == '__main__':
    main()
