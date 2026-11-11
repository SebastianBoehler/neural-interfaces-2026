"""Score SBNX (ledger s12): each SessionBN family standalone and as an extra chain member (w 0.5 / 1 / 2), nested halves."""
import glob, os, sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; D = REPO + "/runs/sbnx/probs"; K = int(os.environ.get("K", 2))
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}
archs = sorted({os.path.basename(f).split("_k")[0][5:] for f in glob.glob(D + "/sbnx_*.npz")})
G1 = [SC.PEOPLE.index(p) for p in SC.G1]; G2 = [SC.PEOPLE.index(p) for p in SC.G2]


def member(a):
    out, n = {}, 0
    for p in A.PROTOS:
        zs = [np.load(f) for f in sorted(glob.glob(f"{D}/sbnx_{a}_k{K}-{p}-s*.npz"))]; n = len(zs)
        assert zs and all((z["index"] == IDX[p]).all() for z in zs), (a, p)
        out[p] = np.mean([z["proba"] for z in zs], 0)
    return out, n


ref = {}
for bn in env.EXPL:
    s = A.summary(A.system(bn, A.FAMS)); ref[bn] = (s["dn"], s["ds"])
print(f"k {K}; reference J", [round(0.5 * v[0].mean() + 0.5 * v[1].mean(), 4) for v in ref.values()])
print("family      seeds | standalone new seen | chain dJ at w 0.5 / 1 / 2 (new, seen)")
res = {}
for a in archs:
    P, n = member(a); sn, ss = SC.person(P["all_loso"], "all_loso"), SC.person(P["seen_session"], "seen_session"); A.EXTRA[a] = P; line = []
    for w in (0.5, 1.0, 2.0):
        dn, ds = [], []
        for bn in env.EXPL:
            s = A.summary(A.system(bn, A.FAMS, extra=[(a, w)])); dn.append(s["dn"] - ref[bn][0]); ds.append(s["ds"] - ref[bn][1])
        dn, ds = np.mean(dn, 0), np.mean(ds, 0); res[(a, w)] = (dn, ds); line.append(f"{0.5 * dn.mean() + 0.5 * ds.mean():+.2f} ({dn.mean():+.2f}, {ds.mean():+.2f})")
    print(f"{a:11s} {n}     | {sn.mean():6.2f} {ss.mean():6.2f} | " + " / ".join(line), flush=True)
half = lambda k, g: 0.5 * res[k][0][g].mean() + 0.5 * res[k][1].mean()  # noqa: E731  (seen people enter both halves)
p1 = max(res, key=lambda k: half(k, G1)); p2 = max(res, key=lambda k: half(k, G2))
print(f"nested: pick on G1 {p1} -> held-out G2 new {res[p1][0][G2].mean():+.2f}; pick on G2 {p2} -> held-out G1 new {res[p2][0][G1].mean():+.2f}; "
      f"nested new {0.5 * (res[p1][0][G2].mean() + res[p2][0][G1].mean()):+.2f}; best overall {max(res, key=lambda k: 0.5 * res[k][0].mean() + 0.5 * res[k][1].mean())}")
