"""recog_lib.Gate.stream == package IdentityGate.step scores (public gallery, real windows)."""
import importlib.util, sys
import numpy as np
sys.path[:0] = ["laneQ", "2026-competition"]
from recog_lib import Gate
pk = sys.argv[1]
spec = importlib.util.spec_from_file_location("pk", pk + "/submission.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
z = dict(np.load(pk + "/weights.npz", allow_pickle=False))
ref, g = mod.IdentityGate(z), Gate(z)
X = np.load("data/proteus_windows.npz", allow_pickle=True)["X"][:40]
ref.reset_state(); worst = 0
Fp = g.proj(np.stack([ref.feat(x) for x in X]))
S = g.stream(g.G0, g.m0, Fp)
for t, x in enumerate(X):
    ref.sum = ref.sum + ref.feat(x); ref.n += 1
    worst = max(worst, np.abs(ref.scores() - S[t]).max())
print("max |diff| over 40 prefixes", worst); assert worst < 1e-9
