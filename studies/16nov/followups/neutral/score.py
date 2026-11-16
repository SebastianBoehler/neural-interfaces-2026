"""Score only the complete registered ENU control and matched §302 repairs."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPAIR = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes302_20261116/repair')
sys.path[:0] = [str(HERE.parent / 'repair'), str(HERE.parent / 'system')]
spec = importlib.util.spec_from_file_location('repair_score_neutral', HERE.parent / 'repair/score.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
s, m, OLD = r.s, r.m, r.OLD
from rolling_audit import paths_for, record


def install():
    r.ARMS['ENU'] = ('neutral', 0.)
    r.install_repair_guard()
    original = s.verify_manifest
    def verify(mf, arm, order, cache, root=None):
        original(mf, arm, order, cache, root)
        if arm == 'ENU' or (arm == 'ZR' and order == 'within0'):
            names = set(mf['inputs_sha256'])
            for suffix in ('/followups/neutral/run.py', '/followups/neutral/transform.py'):
                if not any(x.endswith(suffix) for x in names): raise ValueError('neutral source provenance missing')
    s.verify_manifest = verify


def setup():
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, len(y), sessions)
    install()
    return y, cells, people, old


def zero_guards(old, n):
    m.ARMS.update({'ZR': ('eegn', 0., 'none', 1.), 'TI': ('eegn', .5, 'none', 1.), 'PI': ('eegn', .5, 'none', 1.)})
    s.RECIPES.update({'ZR': ('eegn', 'none', True), 'TI': ('eegn', 'none', False), 'PI': ('eegn', 'none', False)})
    for order in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, np.load(ROOT / f'runs/newmember/s281_v66p_ZR_{order}_gate.npz') as z:
            m.verify(dict(c), dict(z), zero=True)
        for base in s.BASES:
            s.verify_zero(old['control', order, base], s.load_output(ROOT, 'ZR', order, base, n, {}))
    for pattern in ('runs/newmember/s281_v66p_ZR_within0_gate.npz','runs/chainout/s281_v66p_ZR_within0.npz'):
        with np.load(ROOT / pattern) as a, np.load(REPAIR / pattern) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in b.files):
                raise ValueError('fresh zero differs from §302 source')


def main():
    prereg = ROOT / 'runs/lane1_inputs/neutral-registration.md'
    if '§302 neutral-information control' not in prereg.read_text(): raise ValueError('neutral registration missing')
    y, cells, people, old = setup()
    m.ROOT = REPAIR
    m.ARMS = {a: ('eegn', .5, 'none', 1.) for a in ('E05','ET05','ET2','EPR25','EPR50')}
    repairs = m.collect_new(len(y), {})
    m.ROOT = ROOT; m.ARMS = {'ENU': ('eegn', .5, 'none', 1.)}
    neutral = m.collect_new(len(y), {})
    zero_guards(old, len(y))
    for arm in ('TI','PI'):
        for base in s.BASES:
            s.verify_zero(repairs['E05', 'within0', base], s.load_output(ROOT, arm, 'within0', base, len(y), {}))
    def person(arm, order):
        src = old if arm == 'control' else neutral if arm == 'ENU' else repairs
        return np.mean([s.person_accuracy(src[arm,order,b], y, cells, people, s.PEOPLE) for b in s.BASES], axis=0)
    counts = np.array([len(np.unique(cells[people == p])) for p in s.PEOPLE])
    paths = paths_for(ROOT, ['ENU','ZR'], s.ORDERS, s.BASES)
    paths += paths_for(REPAIR, ['E05','ET05','ET2','EPR25','EPR50'], s.ORDERS, s.BASES)
    paths += paths_for(OLD, ['control','zero_eegn'], s.ORDERS, s.BASES)
    paths += paths_for(ROOT, ['TI','PI'], ('within0',), s.BASES)
    paths += [HERE / x for x in ('run.py','transform.py','cp.py','task.py','score.py')]
    paths += [ROOT / 'neutral-reuse.sha256', ROOT / 'neutral-src.sha256']
    audit, sha, previous = record('302', 'ENU_control', paths, prereg, s.sha256)
    print('ENU neutral-information control; exploratory repeated dev selection, not a package claim.')
    print('audit', audit, 'entry_sha256', sha, 'already_read_blocks', previous)
    for arm, ref in [('ENU','control'),('ENU','E05')] + [(a,'ENU') for a in ('ET05','ET2','EPR25','EPR50')]:
        d = np.stack([person(arm,o) - person(ref,o) for o in s.ORDERS])
        print(arm, '-', ref, s.summarize(d,s.ORDERS,s.PEOPLE), 'official_equal_cell', s.cell_weighted_summary(d,s.ORDERS,counts), flush=True)


if __name__ == '__main__': main()
