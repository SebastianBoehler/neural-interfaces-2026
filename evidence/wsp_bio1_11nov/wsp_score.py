"""WSP scoring: verify the cache against the stored core1, then standalone and chain scores of every edit."""
import sys, time
import numpy as np
REPO = "/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026"
sys.path.insert(0, REPO + "/runs/archive_scratch_11nov/secop")
import env
A = env.A; SC = A.SC
Z = np.load(sys.argv[1] + "/edits.npz", allow_pickle=True)
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}
names = sorted({k.split("|")[0] for k in Z.files}); folds = sorted({k.split("|")[1] for k in Z.files})

def assemble(name):
    out = {}
    for p in A.PROTOS:
        G = np.full((2655, 3), np.nan)
        for f in folds:
            if f.startswith(p):
                G[Z[f"{name}|{f}|rows"]] = Z[f"{name}|{f}|proba"]
        out[p] = G[IDX[p]]; assert np.isfinite(out[p]).all()
    return out

base = assemble("base")
for p in A.PROTOS:
    d = np.abs(base[p] - A.C1[p]); print(f"verify {p}: max abs diff {d.max():.2e}, argmax agreement {np.mean(base[p].argmax(1) == A.C1[p].argmax(1)):.4f}")
fams = [f for f in A.FAMS if f != "core1"]
def chain(bn, P):
    A.EXTRA["c1e"] = P; r = A.system(bn, fams, extra=[("c1e", 1.0)], slot="c1e"); s = A.summary(r); return s["dn"], s["ds"]
def stand(P):
    return SC.person(P["all_loso"], "all_loso"), SC.person(P["seen_session"], "seen_session")
G1 = [SC.PEOPLE.index(p) for p in SC.G1]; G2 = [SC.PEOPLE.index(p) for p in SC.G2]
bases = sys.argv[2].split(",") if len(sys.argv) > 2 else list(env.EXPL)
ref = {}
for bn in bases:
    r0 = A.system(bn, A.FAMS); s0 = A.summary(r0); dn, ds = chain(bn, base); ref[bn] = (dn, ds)
    print(f"{bn}: stored-core chain J {0.5 * s0['dn'].mean() + 0.5 * s0['ds'].mean():.4f}; rebuilt-core chain J {0.5 * dn.mean() + 0.5 * ds.mean():.4f}", flush=True)
sn0, ss0 = stand(base); print(f"base standalone new {sn0.mean():.2f} seen {ss0.mean():.2f}")
print("edit         | standalone new  seen | chain dJ   new    G1    G2   seen  (n>=0)")
for nm in names:
    if nm == "base":
        continue
    P = assemble(nm); sn, ss = stand(P); dJ, dnn, dss = [], [], []
    for bn in bases:
        dn, ds = chain(bn, P); dnn.append(dn - ref[bn][0]); dss.append(ds - ref[bn][1])
    dnn, dss = np.mean(dnn, 0), np.mean(dss, 0)
    print(f"{nm:12s} | {sn.mean() - sn0.mean():+6.2f} ({int((sn >= sn0).sum())}/10) {ss.mean() - ss0.mean():+6.2f} | {0.5 * dnn.mean() + 0.5 * dss.mean():+6.2f} {dnn.mean():+6.2f} {dnn[G1].mean():+6.2f} {dnn[G2].mean():+6.2f} {dss.mean():+6.2f}  ({int((dnn >= 0).sum())}/10)", flush=True)
