"""TTT as a final stage: chain output x (fine-tuned net posterior)^lambda, instead of the net being one of ~15 mix members.
Control: the same nets frozen. Exploratory (not pre-registered): reported with people halves.
Usage: python poe.py PROBS_DIR PREFIX"""
import glob, os, sys
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; D, PRE = sys.argv[1], sys.argv[2]
IDX = {p: A.pl.B[f"idx|{p}"].astype(int) for p in A.PROTOS}; G1 = [SC.PEOPLE.index(p) for p in SC.G1]; G2 = [SC.PEOPLE.index(p) for p in SC.G2]
archs = sorted({os.path.basename(f)[len(PRE) + 1:].split("_")[0] for f in glob.glob(f"{D}/{PRE}_*_base-all_loso-s*.npz")})
vars_ = sorted({os.path.basename(f)[len(PRE) + 1:].split("-")[0].split("_", 1)[1] for f in glob.glob(f"{D}/{PRE}_*-all_loso-s*.npz")}, key=lambda v: (v != "base", v))


def expert(v):
    out = {}
    for p in A.PROTOS:
        ms = []
        for a in archs:
            zs = [np.load(f) for f in sorted(glob.glob(f"{D}/{PRE}_{a}_{v}-{p}-s*.npz"))]
            if zs:
                assert all((z["index"] == IDX[p]).all() for z in zs); ms.append(np.mean([z["proba"] for z in zs], 0))
        out[p] = np.exp(np.mean([np.log(np.clip(m, 1e-6, 1)) for m in ms], 0)); out[p] /= out[p].sum(1, keepdims=True)
    return out


base = {bn: A.system(bn, A.FAMS)["out"] for bn in env.EXPL}
def score(P):
    return SC.person(P["all_loso"], "all_loso"), SC.person(P["seen_session"], "seen_session")
ref = [score(base[bn]) for bn in env.EXPL]; dn0 = np.mean([r[0] for r in ref], 0); ds0 = np.mean([r[1] for r in ref], 0)
print(f"archs {archs}; shipped chain J {0.5 * dn0.mean() + 0.5 * ds0.mean():.3f}")
print("expert   lambda | dJ   new (G1, G2, n>=0) seen")
for v in vars_:
    E = expert(v)
    for lam in (0.25, 0.5, 1.0):
        dn, ds = [], []
        for bn in env.EXPL:
            P = {p: base[bn][p] * E[p] ** lam for p in A.PROTOS}; s = score({p: q / q.sum(1, keepdims=True) for p, q in P.items()}); dn.append(s[0]); ds.append(s[1])
        dn, ds = np.mean(dn, 0) - dn0, np.mean(ds, 0) - ds0
        print(f"{v:8s} {lam:5.2f} | {0.5 * dn.mean() + 0.5 * ds.mean():+.2f} {dn.mean():+.2f} ({dn[G1].mean():+.2f}, {dn[G2].mean():+.2f}, {int((dn >= 0).sum())}/10) {ds.mean():+.2f}", flush=True)
