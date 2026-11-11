"""Score FORGET: standalone core1 by half-life (overall and by window position) and through the chain."""
import sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC
Z = np.load(sys.argv[1], allow_pickle=True); IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}
names = sorted({k.split("|")[0] for k in Z.files}, key=lambda s: -float(s[1:])); folds = sorted({k.split("|")[1] for k in Z.files})


def assemble(name):
    out = {}
    for p in A.PROTOS:
        G = np.full((2655, 3), np.nan)
        for f in folds:
            if f.startswith(p):
                G[Z[f"{name}|{f}|rows"]] = Z[f"{name}|{f}|proba"]
        out[p] = G[IDX[p]]; assert np.isfinite(out[p]).all()
    return out


POS = {}
for p in A.PROTOS:
    ss = SC.B[f"sess|{p}"]; pos = np.zeros(len(ss), int)
    for s in np.unique(ss):
        m = np.flatnonzero(ss == s); pos[m] = np.arange(len(m))
    POS[p] = pos
base = assemble(names[0])
for p in A.PROTOS:
    print(f"harness {p}: {names[0]} vs stored core1 max |diff| {np.abs(base[p] - A.C1[p]).max():.2e}")
fams = [f for f in A.FAMS if f != "core1"]; ref = {}
for bn in env.EXPL:
    A.EXTRA["c1e"] = base; s = A.summary(A.system(bn, fams, extra=[("c1e", 1.0)], slot="c1e")); ref[bn] = (s["dn"], s["ds"])
sn0, ss0 = SC.person(base["all_loso"], "all_loso"), SC.person(base["seen_session"], "seen_session")
print("half-life | standalone new seen | accuracy windows 0-10 / 10-90 / 90+ (new; seen) | chain dJ new seen")
for nm in names:
    P = assemble(nm); sn, ss = SC.person(P["all_loso"], "all_loso"), SC.person(P["seen_session"], "seen_session"); b = []
    for p in A.PROTOS:
        y = SC.B[f"y|{p}"]; ok = P[p].argmax(1) == y; b.append(" / ".join(f"{ok[(POS[p] >= lo) & (POS[p] < hi)].mean():.3f}" for lo, hi in ((0, 10), (10, 90), (90, 999))))
    dn, ds = [], []
    for bn in env.EXPL:
        A.EXTRA["c1e"] = P; s = A.summary(A.system(bn, fams, extra=[("c1e", 1.0)], slot="c1e")); dn.append(s["dn"] - ref[bn][0]); ds.append(s["ds"] - ref[bn][1])
    dn, ds = np.mean(dn, 0), np.mean(ds, 0)
    print(f"{nm:9s} | {sn.mean() - sn0.mean():+6.2f} ({int((sn >= sn0).sum())}/10) {ss.mean() - ss0.mean():+6.2f} | {b[0]} ; {b[1]} | {0.5 * dn.mean() + 0.5 * ds.mean():+.2f} {dn.mean():+.2f} {ds.mean():+.2f}", flush=True)
