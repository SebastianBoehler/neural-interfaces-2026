"""Unchanged §281 CP for the fixed ENU and new-source zero arms."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
path = ROOT / 'studies/14nov/newmember/s281_cp.py'
src = path.read_text()
old = "    assert arm in {'control', 'C10S', 'E', 'D', 'zero_eegn', 'C10A', 'C3S', 'C3A'}"
if src.count(old) != 1: raise ValueError('unexpected upstream CP guard')
src = src.replace(old, "    assert arm in {'ENU', 'ZR'}", 1)
anchor = '    tracker = InputTracker()\n'
if src.count(anchor) != 1: raise ValueError('unexpected CP tracker')
src = src.replace(anchor, anchor + f'    tracker.track({str(HERE / "cp.py")!r})\n    tracker.track({str(HERE.parent / "system/recipes.py")!r})\n', 1)
exec(compile(src, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})
