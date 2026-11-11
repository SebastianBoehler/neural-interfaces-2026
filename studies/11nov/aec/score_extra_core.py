"""Score a ts_core2-style member (npz with '<head>|<proto>' and 'index|<proto>') standalone and as an extra chain member."""
import sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; Z = np.load(sys.argv[1]); name = sys.argv[2]
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; G1 = [SC.PEOPLE.index(p) for p in SC.G1]; G2 = [SC.PEOPLE.index(p) for p in SC.G2]
ref = {}
for bn in env.EXPL:
    s = A.summary(A.system(bn, A.FAMS)); ref[bn] = (s["dn"], s["ds"])
for h in ("lr3", "lr10"):
    P = {}
    for p in A.PROTOS:
        assert (Z[f"index|{p}"] == IDX[p]).all(); P[p] = Z[f"{h}|{p}"].astype(np.float64)
    sn, ss = SC.person(P["all_loso"], "all_loso"), SC.person(P["seen_session"], "seen_session"); A.EXTRA[name + h] = P; line = []
    for w in (0.5, 1.0, 2.0):
        dn, ds = [], []
        for bn in env.EXPL:
            s = A.summary(A.system(bn, A.FAMS, extra=[(name + h, w)])); dn.append(s["dn"] - ref[bn][0]); ds.append(s["ds"] - ref[bn][1])
        dn, ds = np.mean(dn, 0), np.mean(ds, 0); line.append(f"w{w}: {0.5 * dn.mean() + 0.5 * ds.mean():+.2f} (new {dn.mean():+.2f}: G1 {dn[G1].mean():+.2f}, G2 {dn[G2].mean():+.2f}; seen {ds.mean():+.2f})")
    print(f"{name} {h}: standalone new {sn.mean():.2f} seen {ss.mean():.2f} | chain " + " | ".join(line), flush=True)
