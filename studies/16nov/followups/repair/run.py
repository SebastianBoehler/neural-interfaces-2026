"""§302 stream repairs within the unchanged §294 full-chain runner."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SYSTEM = HERE.parent / 'system'
ORIGINAL = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes294_20261116/system')
sys.path[:0] = [str(HERE), str(SYSTEM)]
spec = importlib.util.spec_from_file_location('joint_run', SYSTEM / 'run.py')
joint = importlib.util.module_from_spec(spec); spec.loader.exec_module(joint)
ARMS = {'ET05': ('temp', .5), 'ET2': ('temp', 2.), 'EPR25': ('center', .25),
        'EPR50': ('center', .5), 'TI': ('temp', 1.), 'PI': ('center', 0.), 'ZR': ('temp', 1.)}
for name in ARMS:
    joint.recipes.ARMS[name] = ('eegn', 0. if name == 'ZR' else .5, 'none', 1.)
original_load, original_validate, original_patch = joint.load_base, joint.validate, joint.patch_source


def load_base():
    base = original_load(); parent = base.InputTracker
    class Tracker(parent):
        def __init__(self):
            super().__init__(); self.track(HERE / 'run.py'); self.track(HERE / 'transform.py')
    base.InputTracker = Tracker
    return base


def validate(name, add, swap, zero, seeds):
    if name not in ARMS: raise ValueError(f'unregistered repair {name}')
    if name == 'ZR':
        if (add, swap, zero, seeds) != ('eegn', 'none', '1', '0,1,2,3,4,5'):
            raise ValueError('invalid repair zero recipe')
        value = {'name': name, 'add': add, 'swap': swap, 'zero': True, 'seeds': seeds,
                 'eeg_scale': 0., 'coral_mode': 'none', 'fbl_factor': 1.}
    else:
        value = original_validate(name, add, swap, zero, seeds)
    value['repair_mode'], value['repair_value'] = ARMS[name]
    return value


def patch(base, source):
    src = original_patch(base, source)
    old = '            A.EXTRA[f"nm_{fam}"] = {p0: stream(fam, bn)[P], OTHER: np.full((S.N[OTHER], 3), np.nan)}'
    new = '''            from repair_transform import apply, check_causal
            from repair_arms import ARMS as REPAIR_ARMS
            data = stream(fam, bn)[P]
            mode, value = REPAIR_ARMS[NAME]
            sessions = A.L.Z[f"sess|{p0}"]
            changed = check_causal(data, sessions, value) if mode == "center" and value else apply(data, sessions, mode, value)
            if NAME not in ("TI", "PI", "ZR"):
                assert not np.array_equal(changed, data), "repair has no effect"
            A.EXTRA[f"nm_{fam}"] = {p0: changed, OTHER: np.full((S.N[OTHER], 3), np.nan)}'''
    if src.count(old) != 1: raise ValueError('unexpected candidate stream injection')
    return src.replace(old, new, 1)


def verify_identity(name, order):
    if name not in ('TI', 'PI'): return
    root = HERE.parents[3]
    for pattern in ('runs/newmember/s281_v66p_{}_{}_gate.npz', 'runs/chainout/s281_v66p_{}_{}.npz'):
        with np.load(root / pattern.format(name, order)) as actual, np.load(ORIGINAL / pattern.format('E05', order)) as ref:
            if set(actual.files) != set(ref.files) or any(not np.array_equal(actual[k], ref[k]) for k in ref.files):
                raise ValueError(f'{name}: E05 chain identity failed')
    print(f'{name}: complete chain equals archived E05', flush=True)


if __name__ == '__main__':
    import types
    transform_spec = importlib.util.spec_from_file_location('repair_transform', HERE / 'transform.py')
    transform = importlib.util.module_from_spec(transform_spec); transform_spec.loader.exec_module(transform)
    sys.modules['repair_transform'] = transform
    mod = types.ModuleType('repair_arms'); mod.ARMS = ARMS; sys.modules[mod.__name__] = mod
    joint.load_base, joint.validate, joint.patch_source = load_base, validate, patch
    joint.main(); verify_identity(__import__('os').environ['NM_NAME'], sys.argv[1])
