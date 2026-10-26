"""Sort saved probability files into global row order (in place).
Usage: python scripts/sort_probs.py DIR"""
import sys
from pathlib import Path

import numpy as np

for f in sorted(Path(sys.argv[1]).glob("*.npz")):
    z = np.load(f)
    o = np.argsort(z["index"])
    if (o != np.arange(len(o))).any():
        np.savez(f, proba=z["proba"][o], index=z["index"][o])
        print("sorted", f.name)
