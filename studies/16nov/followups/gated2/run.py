"""§296 current-row gate around the immutable §294 full-chain runner."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM = HERE.parent / 'system'
sys.path[:0] = [str(HERE), str(SYSTEM)]
spec = importlib.util.spec_from_file_location('joint_run', SYSTEM / 'run.py')
joint = importlib.util.module_from_spec(spec); spec.loader.exec_module(joint)
ARMS = {'EG06': ('eegn', .6), 'EG08': ('eegn', .8), 'DG06': ('eegn_dist2', .6), 'DG08': ('eegn_dist2', .8), 'ZG06': ('eegn', .6)}
for name, (family, _) in ARMS.items():
    joint.recipes.ARMS[name] = (family, 0. if name == 'ZG06' else 1., 'none', 1.)
original_load, original_validate, original_patch = joint.load_base, joint.validate, joint.patch_source


def load_base():
    base = original_load(); parent = base.InputTracker
    class Tracker(parent):
        def __init__(self):
            super().__init__(); self.track(HERE / 'run.py'); self.track(HERE / 'mask.py')
    base.InputTracker = Tracker
    return base


def validate(name, add, swap, zero, seeds):
    if name not in ARMS:
        raise ValueError(f'unregistered gated arm {name}')
    if name == 'ZG06':
        if (add, swap, zero, seeds) != ('eegn', 'none', '1', '0,1,2,3,4,5'):
            raise ValueError('invalid gated zero recipe')
        value = {'name': name, 'add': add, 'swap': swap, 'zero': True, 'seeds': seeds,
                 'eeg_scale': 0., 'coral_mode': 'none', 'fbl_factor': 1.}
    else:
        value = original_validate(name, add, swap, zero, seeds)
    value['confidence_threshold'] = ARMS[name][1]
    return value


def patch(base, source):
    src = original_patch(base, source)
    anchor = '        d = float(np.abs(m[0] - m0[0]).max())\n'
    # The generated runner executes in its own namespace, so import fixed data directly.
    injected = '''        from mask import apply as apply_gate
        from run_gated_arms import ARMS as GATED_ARMS
        gated, counts = apply_gate(m0[0], m[0], GATED_ARMS[NAME][1], zero=NAME == "ZG06")
        counts["base"] = bn
        GATE_STATS.append(counts)
        m = (gated, m[1])
'''
    if src.count(anchor) != 1:
        raise ValueError('unexpected mix anchor')
    src = src.replace(anchor, injected + anchor, 1)
    anchor = 'CUR, STREAM, LOG = {"P": np.arange(N)}, {}, []'
    if src.count(anchor) != 1:
        raise ValueError('missing stats anchor')
    src = src.replace(anchor, anchor + '\nGATE_STATS = []', 1)
    anchor = 'if FAMS_ or CONFIG[3] != 1.0:\n    assert LOG,'
    if src.count(anchor) != 1:
        raise ValueError('missing final gate stats anchor')
    save = '''import json
assert len(GATE_STATS) == 6 and len({x["base"] for x in GATE_STATS}) == 6, "missing gate counts"
stats_path = f"{REPO}/runs/newmember/s296_{NAME}_{O}_gate_counts.json"
with open(stats_path, "x") as f:
    json.dump({"arm": NAME, "order": O, "counts": GATE_STATS}, f, sort_keys=True)
'''
    return src.replace(anchor, save + anchor, 1)


if __name__ == '__main__':
    # Importable in the source executed by the upstream runner.
    import types
    mod = types.ModuleType('run_gated_arms'); mod.ARMS = ARMS; sys.modules[mod.__name__] = mod
    joint.load_base, joint.validate, joint.patch_source = load_base, validate, patch
    joint.main()
