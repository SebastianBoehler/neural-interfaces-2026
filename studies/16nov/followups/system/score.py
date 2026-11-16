"""Collect only the complete §294 matrix through the existing guarded scorer."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain')
sys.path.insert(0, str(HERE))
from recipes import ARMS, ZERO

path = ROOT / 'studies/14nov/newmember/score_s_v66p.py'
spec = importlib.util.spec_from_file_location('old_score', path)
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
from s281_guard import verify


def expected(name):
    family, _, coral, _ = ZERO if name == 'zero' else ARMS[name]
    add = ([] if family == 'none' else [family]) + (['fb_coral10'] if coral == 'add' else [])
    return (','.join(add) or 'none', 'fb_coral10' if coral == 'swap' else 'none', name == 'zero')


def install_guard():
    original_verify = s.verify_manifest
    s.RECIPES = dict(s.RECIPES, **{a: expected(a) for a in (*ARMS, 'zero')})
    def verify_manifest(m, arm, order, cache, root=None):
        original_verify(m, arm, order, cache, root)
        r = m['recipe']; family, scale, coral, fbl = ZERO if arm == 'zero' else ARMS[arm]
        if (r.get('eeg_scale'), r.get('coral_mode'), r.get('fbl_factor')) != (scale, coral, fbl):
            raise ValueError(f'{arm}: missing or changed weight recipe')
        names = {Path(x).name for x in m['inputs_sha256']}
        if not {'run.py', 'recipes.py'} <= names:
            raise ValueError(f'{arm}: wrapper provenance missing')
    s.verify_manifest = verify_manifest


def collect_new(n, cache):
    install_guard()
    remote = s.remote_manifest(ROOT / 'runs/lane1_inputs/streams.sha256')
    result = {}
    for order in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as z:
            ctl = dict(z)
        for arm in ARMS:
            provenance = ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json'
            m = json.loads(provenance.read_text())
            add, swap, _ = expected(arm)
            for family in [f for group in (add, swap) if group != 'none' for f in group.split(',')]:
                for seed in range(6):
                    name = f'ttt18{order}_{family}_base-all_loso-s{seed}.npz'
                    found = [v for k, v in m['inputs_sha256'].items() if Path(k).name == name]
                    if len(found) != 1 or remote.get(name) != found[0]:
                        raise ValueError(f'{arm}/{order}: stream differs {name}')
            with np.load(ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz') as z:
                verify(ctl, dict(z), zero=False)
            for base in s.BASES:
                result[arm, order, base] = s.load_output(ROOT, arm, order, base, n, cache)
                side = json.loads((ROOT / f'runs/s281_v66_cp/{arm}_{order}_{base}.json').read_text())
                names = {Path(x).name for x in side['inputs_sha256']}
                if not {'cp.py', 'recipes.py'} <= names:
                    raise ValueError(f'{arm}: CP wrapper provenance missing')
    return result


def main():
    if len(sys.argv) != 1:
        raise ValueError('score requires the complete registered matrix')
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, len(y), sessions)
    new = collect_new(len(y), {})
    for order in s.ORDERS:
        with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(ROOT / f'runs/newmember/s281_v66p_zero_{order}_gate.npz') as z:
            verify(dict(c), dict(z), zero=True)
        for base in s.BASES:
            s.verify_zero(old['control', order, base], s.load_output(ROOT, 'zero', order, base, len(y), {}))
    outputs = {**old, **new}
    metric = {(arm, order): np.mean([s.person_accuracy(outputs[arm, order, base], y, cells, people, s.PEOPLE)
                                     for base in s.BASES], axis=0)
              for arm in (*ARMS, 'control', 'E', 'D', 'C10S', 'C10A') for order in s.ORDERS}
    counts = np.array([len(np.unique(cells[people == p])) for p in s.PEOPLE])
    def report(label, terms):
        d = np.stack([sum(weight * metric[name, o] for name, weight in terms) for o in s.ORDERS])
        print(label, s.summarize(d, s.ORDERS, s.PEOPLE), 'cell_equal', s.cell_weighted_summary(d, s.ORDERS, counts), flush=True)
    print('§294 complete matrix: held-out new people; returning route/package remains unvalidated.', flush=True)
    for arm in ARMS:
        report(f'{arm} - control', [(arm, 1), ('control', -1)])
    for arm in ARMS:
        family, scale, coral, fbl = ARMS[arm]
        if coral != 'none':
            single = ('E' if family == 'eegn' else 'D') if scale == 1 else (('E' if family == 'eegn' else 'D') + ('05' if scale == .5 else '2'))
            report(f'{arm} interaction', [(arm, 1), (single, -1), ('C10S' if coral == 'swap' else 'C10A', -1), ('control', 1)])
        elif fbl != 1 and family != 'none':
            report(f'{arm} interaction', [(arm, 1), ('E' if family == 'eegn' else 'D', -1), ('FBL15', -1), ('control', 1)])


if __name__ == '__main__':
    main()
