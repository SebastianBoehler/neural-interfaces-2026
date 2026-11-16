"""Capture original CP's filtered control input before any output write."""
import sys
from pathlib import Path
import numpy as np


class Captured(Exception):
    pass


def patch(source):
    anchors = {
        '    assert not dst.exists() and not inputs.exists(), f"refusing to overwrite {dst} or {inputs}"':
            '    pass  # no output is written in this read-only diagnostic',
        '    result = apply_sessions(windows, raw, filtered, sessions)':
            '    raise Captured(filtered.copy())',
    }
    for before, after in anchors.items():
        if source.count(before) != 1: raise ValueError('unexpected original CP source')
        source = source.replace(before, after, 1)
    return source


def original_filtered(old_root, order, base):
    path = Path(old_root) / 'studies/14nov/newmember/s281_cp.py'
    source = patch(path.read_text())
    previous_argv, previous_path, load = sys.argv, sys.path[:], np.load
    sys.argv = [str(path), 'control', order, base]
    try:
        try:
            exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path), 'Captured': Captured})
        except Captured as result:
            return result.args[0]
        raise ValueError('original CP did not reach input capture')
    finally:
        sys.argv, sys.path[:], np.load = previous_argv, previous_path, load
