"""Guard and score one complete §302 repair against control and E05."""
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
from rolling_audit import paths_for, record


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('ET05','ET2','EPR25','EPR50'):
        raise ValueError('one fixed treatment arm required')
    arm = sys.argv[1]; prereg = ROOT / 'runs/lane1_inputs/repair-registration.md'
    if not prereg.is_file() or '### 302.' not in prereg.read_text() or '### 298.' not in prereg.read_text():
        raise ValueError('§298/302 registration missing')
    required = [ROOT / f'runs/s281_v66_cp/{arm}_{o}_{b}.npz' for o in s.ORDERS for b in s.BASES]
    if not all(p.is_file() and p.with_suffix('.json').is_file() for p in required):
        raise ValueError('complete eight-order, six-base treatment required')
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, len(y), sessions)
    r.install_repair_guard(); m.ARMS = {a: ('eegn', .5, 'none', 1.) for a in ('E05', arm)}
    new = m.collect_new(len(y), {})
    m.ARMS.update({'ZR': ('eegn', 0., 'none', 1.), 'TI': ('eegn', .5, 'none', 1.), 'PI': ('eegn', .5, 'none', 1.)})
    s.RECIPES.update({'ZR': ('eegn', 'none', True), 'TI': ('eegn', 'none', False), 'PI': ('eegn', 'none', False)})
    for order in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(ROOT / f'runs/newmember/s281_v66p_ZR_{order}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for b in s.BASES: s.verify_zero(old['control', order, b], s.load_output(ROOT, 'ZR', order, b, len(y), {}))
    for identity in ('TI','PI'):
        for b in s.BASES:
            s.verify_zero(new['E05', 'within0', b], s.load_output(ROOT, identity, 'within0', b, len(y), {}))
    def person(a, order):
        source = old if a == 'control' else new
        return np.mean([s.person_accuracy(source[a, order, b], y, cells, people, s.PEOPLE) for b in s.BASES], axis=0)
    counts = np.array([len(np.unique(cells[people == p])) for p in s.PEOPLE])
    paths = paths_for(ROOT, [arm, 'E05', 'ZR'], s.ORDERS, s.BASES)
    paths += paths_for(OLD, ['control', 'zero_eegn'], s.ORDERS, s.BASES)
    paths += paths_for(ROOT, ['TI', 'PI'], ('within0',), s.BASES)
    paths += [HERE / name for name in ('run.py', 'transform.py', 'task_v3.py', 'cp_v3.py', 'score.py', 'rolling.py')]
    audit, sha, previous = record('302', arm, paths, prereg, s.sha256)
    print('Sequential dev read; full repair wave and saved-stage diagnostics still required.')
    print('audit', audit, 'entry_sha256', sha, 'already_read_blocks', previous)
    for reference in ('control', 'E05'):
        d = np.stack([person(arm, o) - person(reference, o) for o in s.ORDERS])
        print(arm, '-', reference, s.summarize(d, s.ORDERS, s.PEOPLE), 'official_equal_cell',
              s.cell_weighted_summary(d, s.ORDERS, counts), flush=True)


if __name__ == '__main__': main()
