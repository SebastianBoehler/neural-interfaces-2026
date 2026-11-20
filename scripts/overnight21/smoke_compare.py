"""Zero-change check of the guard: smoke_pred (hardcb_safe, order 71000, P01) vs stored production hardcb pred for the same order."""
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
keys = [k for k in a.files if k.endswith("_71000_1") or "_71000_" in k]
assert keys, a.files
tot = diff = 0
for k in keys:
    if k.startswith("secs"):
        continue
    assert k in b.files, k
    x, y = a[k], b[k]
    assert x.shape == y.shape, k
    tot += x.size; diff += int((x != y).sum())
    print(k, x.size, "changed", int((x != y).sum()))
print("TOTAL values", tot, "changed", diff)
assert diff == 0, "guard changed decisions or delivery"
print("ZERO-CHANGE OK")
