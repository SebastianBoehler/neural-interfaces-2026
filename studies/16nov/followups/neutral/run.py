"""Run the registered ENU control through the original §302 chain."""
import importlib.util
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent / 'repair'
sys.path[:0] = [str(HERE), str(REPAIR)]
spec = importlib.util.spec_from_file_location('repair_run_neutral', REPAIR / 'run.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
r.ARMS['ENU'] = ('neutral', 0.)
r.joint.recipes.ARMS['ENU'] = ('eegn', .5, 'none', 1.)
original_load, original_patch = r.load_base, r.patch


def load_base():
    base = original_load(); parent = base.InputTracker
    class Tracker(parent):
        def __init__(self):
            super().__init__(); self.track(HERE / 'run.py'); self.track(HERE / 'transform.py')
    base.InputTracker = Tracker
    return base


def patch(base, source):
    src = original_patch(base, source)
    old = '            changed = check_causal(data, sessions, value) if mode == "center" and value else apply(data, sessions, mode, value)'
    new = '''            from neutral_transform import apply as neutral
            changed = neutral(data) if NAME == "ENU" else (check_causal(data, sessions, value) if mode == "center" and value else apply(data, sessions, mode, value))
            if NAME == "ENU":
                assert np.array_equal(changed, np.full_like(data, 1. / 3.)), "neutral slot is not uniform"'''
    if src.count(old) != 1: raise ValueError('unexpected §302 transform injection')
    return src.replace(old, new, 1)


if __name__ == '__main__':
    for module_name, path in (('repair_transform', REPAIR / 'transform.py'), ('neutral_transform', HERE / 'transform.py')):
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(mod); sys.modules[module_name] = mod
    arms = types.ModuleType('repair_arms'); arms.ARMS = r.ARMS; sys.modules[arms.__name__] = arms
    r.joint.load_base, r.joint.validate, r.joint.patch_source = load_base, r.validate, patch
    r.joint.main()
