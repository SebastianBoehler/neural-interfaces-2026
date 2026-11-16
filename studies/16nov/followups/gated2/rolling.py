"""Score one complete §296 arm using the existing guarded scorer primitives."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gated_score', HERE / 'score.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s, root, old = m.s, m.ROOT, m.OLD
from rolling_audit import paths_for, record


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in m.ARMS:
        raise ValueError('supply one fixed registered gated arm')
    arm = sys.argv[1]
    prereg = root / 'runs/lane1_inputs/rolling-registration.md'
    if not prereg.is_file() or '### 298.' not in prereg.read_text():
        raise ValueError('rolling §298 registration required before scoring')
    paths = [root / f'runs/s281_v66_cp/{arm}_{o}_{b}.npz' for o in s.ORDERS for b in s.BASES]
    if not all(p.is_file() and p.with_suffix('.json').is_file() for p in paths):
        raise ValueError(f'{arm}: complete eight-order, six-base output required')
    with np.load(root / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        y, cells, people = z['y|all_loso'], z['cell|all_loso'], z['person|all_loso']
        sessions = len(np.unique(z['sess|all_loso']))
    old_results = s.collect(old, len(y), sessions)
    m.install_guard(); remote = s.remote_manifest(root / 'runs/lane1_inputs/streams.sha256'); cache = {}
    # Require the independent §294 zero/control across all eight orders too.
    system_root = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes294_20261116/system')
    s.RECIPES['zero'] = ('eegn', 'none', True)
    for order in s.ORDERS:
        with np.load(old / f'runs/newmember/s281_v66p_control_{order}_gate.npz') as c, \
             np.load(system_root / f'runs/newmember/s281_v66p_zero_{order}_gate.npz') as z:
            m.joint.verify(dict(c), dict(z), zero=True)
        provenance = json.loads((system_root / f'runs/newmember/s281_v66p_zero_{order}_provenance.json').read_text())
        for seed in range(6):
            name = f'ttt18{order}_eegn_base-all_loso-s{seed}.npz'
            found = [v for k, v in provenance['inputs_sha256'].items() if Path(k).name == name]
            if len(found) != 1 or remote.get(name) != found[0]:
                raise ValueError(f'{order}: zero stream lock differs')
        for base in s.BASES:
            system_zero = s.load_output(system_root, 'zero', order, base, len(y), cache)
            s.verify_zero(old_results['control', order, base], system_zero)
    for order in s.ORDERS:
        zero = m.checked('ZG06', order, len(y), cache, remote)
        for base in s.BASES:
            s.verify_zero(old_results['control', order, base], zero[base])
    deltas = {'control': [], 'E' if arm.startswith('E') else 'D': []}
    for order in s.ORDERS:
        values = m.checked(arm, order, len(y), cache, remote)
        candidate = np.mean([s.person_accuracy(values[b], y, cells, people, s.PEOPLE) for b in s.BASES], axis=0)
        for reference in deltas:
            orig = np.mean([s.person_accuracy(old_results[reference, order, b], y, cells, people, s.PEOPLE)
                            for b in s.BASES], axis=0)
            deltas[reference].append(candidate - orig)
    counts = np.array([len(np.unique(cells[people == p])) for p in s.PEOPLE])
    reference = 'E' if arm.startswith('E') else 'D'
    paths = paths_for(root, [arm, 'ZG06'], s.ORDERS, s.BASES)
    paths += paths_for(system_root, ['zero'], s.ORDERS, s.BASES) + paths_for(old, ['control', 'zero_eegn', reference], s.ORDERS, s.BASES)
    paths += [root / f'runs/newmember/s296_{a}_{o}_gate_counts.json' for a in (arm, 'ZG06') for o in s.ORDERS]
    audit, entry_sha, previous = record('296', arm, paths, prereg, s.sha256)
    print('Sequential dev read; repeated selection; full-matrix report still required.')
    print('audit', audit, 'entry_sha256', entry_sha, 'already_read_blocks', previous, flush=True)
    for reference, delta in deltas.items():
        print(f'{arm} - {reference}', s.summarize(np.stack(delta), s.ORDERS, s.PEOPLE), 'official_equal_cell',
              s.cell_weighted_summary(np.stack(delta), s.ORDERS, counts), flush=True)
    for order in s.ORDERS:
        path = root / f'runs/newmember/s296_{arm}_{order}_gate_counts.json'
        print('gate counts', order, json.loads(path.read_text())['counts'], flush=True)


if __name__ == '__main__': main()
