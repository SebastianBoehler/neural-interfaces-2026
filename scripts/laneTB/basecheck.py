"""Compare the production fold packages' returning-path arrays with the uploaded all-data cap250 (Fable F1 parts)."""
import json
import sys
from pathlib import Path

import numpy as np

ref = np.load(sys.argv[1])
folds = Path(sys.argv[2])
out = {}
for fold in sorted(p.name for p in folds.iterdir() if (p / "pkg/weights.npz").exists()):
    w = np.load(folds / fold / "pkg/weights.npz")
    out[fold] = {k: ("MISSING" if k not in w.files else
                     "IDENTICAL" if w[k].shape == ref[k].shape and np.array_equal(w[k], ref[k]) else "differs")
                 for k in ref.files}
print(json.dumps(out, indent=1))
same = {f: [k for k, v in d.items() if v == "IDENTICAL"] for f, d in out.items()}
print("identical to uploaded:", {f: v for f, v in same.items() if v})
