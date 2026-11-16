"""Run existing arm-specific v6 personalisation with geometry probability guards."""
import os
import sys
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'expected exactly one source anchor: {old}')
    return source.replace(old, new, 1)


def patch_source(source):
    old = 'assert arm in {"control", "zero_coral", "S1", "S2", "S3", "S4", "S5"}'
    new = 'assert arm in {"control", "geo_zero", "geo_s50", "geo_d25", "geo_d50", "geo_d100"}'
    source = replace_once(source, old, new)
    source = replace_once(source, '    tracker = InputTracker()',
                          '    tracker = InputTracker()\n    tracker.track(Path(os.environ["S295_CP_WRAPPER"]))')
    source = replace_once(source, '    assert raw.shape == (K.C.N, 3)',
                          '    assert raw.shape == (K.C.N, 3)\n    assert np.isfinite(raw).all() and (raw >= 0).all() and np.allclose(raw.sum(1), 1), "invalid chain probability"')
    old_filter = '    filtered = K.gate(raw, K.J.filt_joint(raw, hc, "urn"), order, lat)'
    source = replace_once(source, old_filter,
                          old_filter + '\n    assert np.isfinite(filtered).all() and (filtered >= 0).all() and np.allclose(filtered.sum(1), 1), "invalid filtered probability"')
    return source


def main():
    root = Path(os.environ['NI26_WS']).resolve(strict=True)
    path = root / 'studies/14nov/newmember/s281_cp.py'
    os.environ['S295_CP_WRAPPER'] = str(Path(__file__).resolve())
    context = {'__name__': '__main__', '__file__': str(path), 'os': os}
    exec(compile(patch_source(path.read_text()), str(path), 'exec'), context)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('usage: cp.py ARM ORDER BASE')
    main()
