"""Capture the actual arguments of the verified §430 three-seed personalization."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'aux430.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'b60d5da76afe33e3cdf917cf4108adbfd9413d90d4b992c25041d0df4878339f'
spec = importlib.util.spec_from_file_location('aux430', SOURCE)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
order = owner.ORDER
expected = json.loads((ROOT / 'aux430_result.json').read_text())['three_sha256'][order]
archive = ROOT / 'stock' / f'{order}.npz'
assert owner.digest(archive) == expected
captures = []


def capture(frame, event, value):
    if event != 'return' or frame.f_code.co_name != 'apply_sessions':
        return
    if Path(frame.f_code.co_filename).name != 's281_cp.py':
        return
    parent = frame.f_back
    assert parent.f_code is owner.main.__code__
    if parent.f_locals['count'] == 3:
        with np.load(archive) as pinned:
            assert np.array_equal(value, pinned['scores'])
        captures.append(dict(raw=parent.f_locals['value'], filtered=parent.f_locals['filtered'],
                             terminal=value))


sys.setprofile(capture)
try:
    owner.main()
finally:
    sys.setprofile(None)
assert len(captures) == 1
for key, arr in captures[0].items():
    assert arr.shape == (2655, 3) and np.isfinite(arr).all(), key
path = ROOT / f'stages_{order}.npz'
np.savez_compressed(path, **captures[0])
path.with_suffix('.json').write_text(json.dumps(dict(order=order, source=owner.digest(SOURCE),
    output=owner.digest(path), archived_stock=expected, experiment=owner.digest(ROOT/'source.sha256'))))
