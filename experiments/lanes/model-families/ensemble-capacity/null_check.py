"""Lane E identity check on the cluster: E0 (cap250 code path + guard key) must reproduce cap250's labels exactly.

python null_check.py OUT_DIR TAG_E0 TAG_CAP250   compares every p*_c*.npz of TAG_E0 with the same file of TAG_CAP250
(labels, rows, and the recognition / routing meta columns; time column excluded). Exit 1 on any difference or a missing file.
"""
import sys
from pathlib import Path

import numpy as np

out, a, b = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
files = sorted((out / a).glob("p*_c*.npz"))
assert files, f"no {a} files"
n = bad = 0
for f in files:
    g = out / b / f.name
    assert g.exists(), f"{g} missing"
    with np.load(f) as x, np.load(g) as y:
        assert sorted(x.files) == sorted(y.files)
        for k in x.files:
            n += 1
            same = np.array_equal(x[k][:3], y[k][:3]) if k.endswith("|meta") else np.array_equal(x[k], y[k])
            bad += not same
            if not same:
                print("DIFF", f.name, k)
print(f"NULL {'OK' if not bad else 'FAIL'}: {len(files)} files, {n} arrays, {bad} differ")
sys.exit(1 if bad else 0)
