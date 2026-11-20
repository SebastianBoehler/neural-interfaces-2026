"""feat == package IdentityGate.feat on real PROTEUS windows (zero difference required)."""
import importlib.util, sys
import numpy as np
sys.path.insert(0, "laneQ"); sys.path.insert(0, "2026-competition")
from extract_feat import feat
pk = sys.argv[1]
spec = importlib.util.spec_from_file_location("pk", pk + "/submission.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
z = dict(np.load(pk + "/weights.npz", allow_pickle=False))
g = m.IdentityGate(z)
X = np.load("data/proteus_windows.npz", allow_pickle=True)["X"][:20]
d = max(np.abs(g.feat(x) - feat(x)).max() for x in X)
print("max |diff|", d); assert d == 0
