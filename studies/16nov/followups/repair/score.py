"""Score complete §302 full-system repairs with the original guarded scorer."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SYSTEM = HERE.parent / 'system'
sys.path[:0] = [str(HERE), str(SYSTEM)]
from rolling_audit import paths_for, record
spec = importlib.util.spec_from_file_location('joint_score', SYSTEM / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, OLD = m.s, m.OLD
ARMS = {'ET05': ('temp', .5), 'ET2': ('temp', 2.), 'EPR25': ('center', .25), 'EPR50': ('center', .5),
        'TI': ('temp', 1.), 'PI': ('center', 0.), 'ZR': ('temp', 1.)}


def install_repair_guard():
    original = s.verify_manifest
    def verify(mf, arm, order, cache, root=None):
        original(mf, arm, order, cache, root)
        if arm not in ARMS: return
        recipe = mf['recipe']
        if (recipe.get('repair_mode'), recipe.get('repair_value')) != ARMS[arm]:
            raise ValueError(f'{arm}: incorrect repair recipe')
        names = set(mf['inputs_sha256'])
        if not any(x.endswith('/followups/repair/run.py') for x in names) or not any(x.endswith('/followups/repair/transform.py') for x in names):
            raise ValueError(f'{arm}: missing repair source provenance')
    s.verify_manifest = verify


def main():
    prereg = ROOT / 'runs/lane1_inputs/repair-registration.md'
    if not prereg.is_file() or '### 302.' not in prereg.read_text(): raise ValueError('§302 registration required')
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, len(y), sessions)
    install_repair_guard()
    m.ARMS = {a: m.ARMS[a] for a in ('E05',)}
    m.ARMS.update({a: ('eegn', .5, 'none', 1.) for a in ('ET05', 'ET2', 'EPR25', 'EPR50')})
    new = m.collect_new(len(y), {})
    m.ARMS.update({'ZR': ('eegn', 0., 'none', 1.), 'TI': ('eegn', .5, 'none', 1.), 'PI': ('eegn', .5, 'none', 1.)})
    s.RECIPES.update({'ZR': ('eegn', 'none', True), 'TI': ('eegn', 'none', False), 'PI': ('eegn', 'none', False)})
    for order in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(ROOT / f'runs/newmember/s281_v66p_ZR_{order}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for base in s.BASES:
            s.verify_zero(old['control', order, base], s.load_output(ROOT, 'ZR', order, base, len(y), {}))
    for arm in ('TI', 'PI'):
        for base in s.BASES:
            identity = s.load_output(ROOT, arm, 'within0', base, len(y), {})
            s.verify_zero(new['E05', 'within0', base], identity)
    def person(arm, order):
        src = old if arm == 'control' else new
        return np.mean([s.person_accuracy(src[arm, order, b], y, cells, people, s.PEOPLE) for b in s.BASES], axis=0)
    counts = np.array([len(np.unique(cells[people == p])) for p in s.PEOPLE])
    paths = paths_for(ROOT, ['E05','ET05','ET2','EPR25','EPR50','ZR'], s.ORDERS, s.BASES)
    paths += paths_for(OLD, ['control','zero_eegn'], s.ORDERS, s.BASES)
    paths += paths_for(ROOT, ['TI','PI'], ('within0',), s.BASES)
    paths += [HERE / name for name in ('run.py','transform.py','task_v3.py','cp_v3.py','cp.py','score.py')]
    audit, sha, previous = record('302', 'complete_repair_wave', paths, prereg, s.sha256)
    print('§302 exploratory full-system repair; repeated dev selection; no package claim.')
    print('audit', audit, 'entry_sha256', sha, 'already_read_blocks', previous)
    for arm in ('ET05', 'ET2', 'EPR25', 'EPR50'):
        for reference in ('control', 'E05'):
            d = np.stack([person(arm, o) - person(reference, o) for o in s.ORDERS])
            print(arm, '-', reference, s.summarize(d, s.ORDERS, s.PEOPLE),
                  'official_equal_cell', s.cell_weighted_summary(d, s.ORDERS, counts), flush=True)


if __name__ == '__main__': main()
