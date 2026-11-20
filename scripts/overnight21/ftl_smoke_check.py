"""Lane FT-labels no-op guard: the flipped arm MUST change labels. usage: ftl_smoke_check.py FLIPPED.npz UNFLIPPED.npz
Compares every labels_<order>_<session> array; asserts same rows (delivery) and changed > 0 (zero-change is the failure here)."""
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
tot = diff = 0
for k in a.files:
    if k.startswith("rows_"):
        assert np.array_equal(a[k], b[k]), f"delivery differs {k}"
    elif k.startswith("labels_") and k in b.files:
        assert a[k].shape == b[k].shape, k
        n = int((a[k] != b[k]).sum()); tot += a[k].size; diff += n
        print(k, a[k].size, "changed", n)
print("TOTAL labels", tot, "changed", diff, f"({100 * diff / max(tot, 1):.2f} %)")
assert tot > 0, "no labels compared"
assert diff > 0, "ZERO CHANGE: v6_ft_labels flip is a silent no-op"
print("FLIP REACHES THE CODE PATH (labels differ) OK")
