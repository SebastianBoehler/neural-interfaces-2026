"""Use the existing causal-personalization stage with the registered seed arms."""

import os
import sys
from pathlib import Path

root = Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(root / 'studies/16nov/followups/geometry')]
from chain_308 import ARMS  # noqa: E402
from cp import patch_source, replace_once  # noqa: E402

path = root / 'studies/14nov/newmember/s281_cp.py'
source = patch_source(path.read_text())
old = 'assert arm in {"control", "geo_zero", "geo_s50", "geo_d25", "geo_d50", "geo_d100"}'
source = replace_once(source, old, f'assert arm in {set(ARMS)!r}')
anchor = '    tracker.track(Path(os.environ["S295_CP_WRAPPER"]))'
source = replace_once(source, anchor, anchor + f'\n    tracker.track({str(root / "studies/16nov/followups/geometry/cp.py")!r})')
os.environ['S295_CP_WRAPPER'] = str(Path(__file__).resolve())
exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path), 'os': os})
