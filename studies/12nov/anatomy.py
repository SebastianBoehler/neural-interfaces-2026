"""Per-session anatomy of the v3 chain on one base: new (LOSO) and seen sessions; early vs late windows."""
import sys, os, numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay"]
import env
A = env.A; SC = A.SC
bn = sys.argv[1] if len(sys.argv) > 1 else "B2_pool"
out = A.system(bn, A.FAMS)
out = out["out"]
for p in A.PROTOS:
    s = SC.session(out[p], p); print(p, " ".join(f"{k}:{v:.1f}" for k, v in s.items()))
    y, ss = SC.B[f"y|{p}"], SC.B[f"sess|{p}"]; pred = out[p].argmax(1); t = np.zeros(len(y), int)
    for u in np.unique(ss):
        m = np.flatnonzero(ss == u); t[m] = np.arange(len(m))
    for a, b in ((0, 30), (30, 60), (60, 100), (100, 150), (150, 250)):
        m = (t >= a) & (t < b); print(f"   windows {a:3d}-{b:3d}: acc {100 * (pred[m] == y[m]).mean():.1f} (n {m.sum()})")
