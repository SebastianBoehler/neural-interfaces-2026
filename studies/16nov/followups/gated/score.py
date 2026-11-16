"""Score complete §296 matrix with the original guarded §292A primitives."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SYSTEM = HERE.parent / 'system'
sys.path.insert(0, str(SYSTEM))
spec = importlib.util.spec_from_file_location('joint_score', SYSTEM / 'score.py')
joint = importlib.util.module_from_spec(spec); spec.loader.exec_module(joint)
s, OLD = joint.s, joint.OLD
ARMS = {'EG06': ('eegn', .6), 'EG08': ('eegn', .8), 'DG06': ('eegn_dist2', .6), 'DG08': ('eegn_dist2', .8)}


def install_guard():
    original = s.verify_manifest
    s.RECIPES = dict(s.RECIPES, zero=('eegn', 'none', True),
                     **{name: (family, 'none', False) for name, (family, _) in ARMS.items()})
    def verify_manifest(m, arm, order, cache, root=None):
        original(m, arm, order, cache, root)
        recipe = m['recipe']
        if arm in ARMS:
            if (recipe.get('eeg_scale'), recipe.get('coral_mode'), recipe.get('fbl_factor'), recipe.get('confidence_threshold')) != (1., 'none', 1., ARMS[arm][1]):
                raise ValueError(f'{arm}: incorrect gated recipe')
            names = set(recipe_path for recipe_path in m['inputs_sha256'])
            if not any(x.endswith('/followups/gated/run.py') for x in names) or not any(x.endswith('/followups/gated/mask.py') for x in names):
                raise ValueError(f'{arm}: gated source missing')
    s.verify_manifest = verify_manifest


def checked(arm, order, n, cache, remote):
    with np.load(OLD / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
         np.load(ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz') as z:
        joint.verify(dict(c), dict(z), zero=arm == 'zero')
    p = json.loads((ROOT / f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json').read_text())
    family = 'eegn' if arm == 'zero' else ARMS[arm][0]
    for seed in range(6):
        name = f'ttt18{order}_{family}_base-all_loso-s{seed}.npz'
        found = [v for k, v in p['inputs_sha256'].items() if Path(k).name == name]
        if len(found) != 1 or found[0] != remote.get(name):
            raise ValueError(f'{arm}/{order}: mismatched input {name}')
    if arm != 'zero':
        stats = json.loads((ROOT / f'runs/newmember/s296_{arm}_{order}_gate_counts.json').read_text())
        counts = stats.get('counts', [])
        if stats.get('arm') != arm or stats.get('order') != order or {x['base'] for x in counts} != set(s.BASES) or len(counts) != 6:
            raise ValueError(f'{arm}/{order}: incomplete gate counts')
        for x in counts:
            if (x['threshold'] != ARMS[arm][1] or not 0 < x['changed'] <= x['eligible'] <= x['total']
                    or x['ineligible_unchanged'] != x['total'] - x['eligible']):
                raise ValueError(f'{arm}/{order}: invalid gate counts')
    outputs = {}
    for base in s.BASES:
        outputs[base] = s.load_output(ROOT, arm, order, base, n, cache)
        if arm != 'zero':
            side = json.loads((ROOT / f'runs/s281_v66_cp/{arm}_{order}_{base}.json').read_text())
            if not any(x.endswith('/followups/gated/cp.py') for x in side['inputs_sha256']):
                raise ValueError(f'{arm}: gated CP source missing')
    return outputs


def main():
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cell, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old = s.collect(OLD, len(y), sessions)
    install_guard(); remote = s.remote_manifest(ROOT / 'runs/lane1_inputs/streams.sha256'); cache = {}
    zero = checked('zero', 'within0', len(y), cache, remote)
    for base in s.BASES:
        s.verify_zero(old['control', 'within0', base], zero[base])
    metric = {}
    for order in s.ORDERS:
        for arm in ARMS:
            values = checked(arm, order, len(y), cache, remote)
            metric[arm, order] = np.mean([s.person_accuracy(values[b], y, cell, people, s.PEOPLE) for b in s.BASES], axis=0)
        for arm in ('control', 'E', 'D'):
            metric[arm, order] = np.mean([s.person_accuracy(old[arm, order, b], y, cell, people, s.PEOPLE) for b in s.BASES], axis=0)
    counts = np.array([len(np.unique(cell[people == p])) for p in s.PEOPLE])
    print('§296 new people, shuffled delivery, fold-routing emulation; returning/package route not validated')
    for order in s.ORDERS:
        for arm in ARMS:
            print('gate counts', arm, order, json.loads((ROOT / f'runs/newmember/s296_{arm}_{order}_gate_counts.json').read_text())['counts'], flush=True)
    for arm in ARMS:
        for ref in ('control', 'E' if arm.startswith('E') else 'D'):
            delta = np.stack([metric[arm, o] - metric[ref, o] for o in s.ORDERS])
            print(f'{arm} - {ref}', s.summarize(delta, s.ORDERS, s.PEOPLE), 'cell_equal',
                  s.cell_weighted_summary(delta, s.ORDERS, counts), flush=True)


if __name__ == '__main__':
    main()
