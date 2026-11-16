"""Run the existing package-matched CP for one fixed §296 arm."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARMS = ('EG06', 'EG08', 'DG06', 'DG08', 'ZG06')
path = ROOT / 'studies/14nov/newmember/s281_cp.py'
src = path.read_text()
old = "    assert arm in {'control', 'C10S', 'E', 'D', 'zero_eegn', 'C10A', 'C3S', 'C3A'}"
if src.count(old) != 1:
    raise ValueError('unexpected upstream CP guard')
src = src.replace(old, f'    assert arm in {set(ARMS)!r}', 1)
anchor = '    tracker = InputTracker()\n'
if src.count(anchor) != 1:
    raise ValueError('unexpected upstream CP tracker')
src = src.replace(anchor, anchor + f'    tracker.track({str(HERE / "cp.py")!r})\n', 1)
exec(compile(src, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})
