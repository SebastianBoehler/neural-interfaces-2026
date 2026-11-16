"""§294 fixed joint recipes, using the original §281 chain and gate guards."""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import recipes


def load_base():
    path = ROOT / 'studies/14nov/newmember/run_s_v66.py'
    spec = importlib.util.spec_from_file_location('lane294_chain', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(name, add, swap, zero, seeds):
    family, scale, coral, fbl = recipes.get(name)
    adds = ([] if family == 'none' else [family])
    if coral == 'add':
        adds.append('fb_coral10')
    expected = (','.join(adds) or 'none', 'fb_coral10' if coral == 'swap' else 'none', name == 'zero')
    if (add, swap, zero == '1') != expected or zero not in ('', '0', '1') or seeds != '0,1,2,3,4,5':
        raise ValueError(f'{name}: invalid recipe or seeds')
    return {'name': name, 'add': add, 'swap': swap, 'zero': name == 'zero', 'seeds': seeds,
            'eeg_scale': scale, 'coral_mode': coral, 'fbl_factor': fbl}


def patch_source(base, src):
    rep = base.replace_once
    original_assert = 'assert len(SW) <= 1 and len(set(ADD)) == len(ADD) and not (ADD and SW) and all(f in ("eegn", "eegn_dist2", "fb_coral3", "fb_coral10") for f in ADD + SW), (ADD, SW)'
    src = rep(src, original_assert, 'assert len(SW) <= 1 and len(set(ADD + SW)) == len(ADD + SW)')
    src = rep(src, 'if FAMS_:\n    _mix = A.mix', 'if FAMS_ or CONFIG[3] != 1.0:\n    _mix = A.mix')
    src = rep(src, 'if FAMS_:\n    assert LOG,', 'if FAMS_ or CONFIG[3] != 1.0:\n    assert LOG,')
    src = rep(src, 'CUR, STREAM, LOG = {"P": np.arange(N)}, {}, []',
              'from recipes import get, modify_extra\nCONFIG = get(NAME)\nCUR, STREAM, LOG = {"P": np.arange(N)}, {}, []')
    start = '        extra = list(extra); sw = {n: w for n, w in extra if n in ("sc_fbraw", "sc_fb", "sc_sinc", "sc_fbl")}\n'
    end = '        return m\n    A.mix = mix'
    if src.count(start) != 1 or src.count(end) != 1:
        raise ValueError('unexpected upstream mix implementation')
    a, rest = src.split(start, 1)
    _, b = rest.split(end, 1)
    replacement = '''        extra = list(extra)
        for fam in FAMS_:
            A.EXTRA[f"nm_{fam}"] = {p0: stream(fam, bn)[P], OTHER: np.full((S.N[OTHER], 3), np.nan)}
        m = _mix(bn, p, fams, modify_extra(extra, CONFIG), *a, **k)
        family, scale, coral, fbl = CONFIG
        variants = []
        if family != "none" and scale:
            variants.append(("EEG", ("none", 0., coral, fbl)))
        if coral != "none":
            variants.append(("CORAL", (family, scale, "none", fbl)))
        if fbl != 1.:
            variants.append(("FBL", (family, scale, coral, 1.)))
        for knob, variant in variants:
            ref = _mix(bn, p, fams, modify_extra(extra, variant), *a, **k)
            assert not np.array_equal(m[0], ref[0]), f"{knob} knob has no effect"
        d = float(np.abs(m[0] - m0[0]).max())
        if ZERO:
            assert np.array_equal(m[0], m0[0]) and m[1] == m0[1], "zero changed mix"
        else:
            assert d > 0, "recipe does not change the mix"
        assert np.isfinite(m[0]).all(), "non-finite mix"
        LOG.append((bn, scale, d))
        return m
    A.mix = mix'''
    return base.patch_runner(a + replacement + b)


def main():
    base = load_base()
    original_patch = base.patch_runner
    original_tracker = base.InputTracker

    class Tracker(original_tracker):
        def __init__(self):
            super().__init__()
            self.track(__file__)
            self.track(HERE / 'recipes.py')

    base.InputTracker = Tracker
    base.validate_recipe = validate
    class Original:
        replace_once = staticmethod(base.replace_once)
        patch_runner = staticmethod(original_patch)
    base.patch_runner = lambda source: patch_source(Original, source)
    base.main()


if __name__ == '__main__':
    main()
