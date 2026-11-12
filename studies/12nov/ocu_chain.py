"""OCUNET in the chain (ledger s58, step 3): as a mix member, as a final stage, and as part of SessionEM's view.
Usage: python studies/12nov/ocu_chain.py [mix|nomix]"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z; arm = sys.argv[1] if len(sys.argv) > 1 else "nomix"; D = REPO + "/runs/ocunet/probs"
IDX = {p: Z[f"idx|{p}"].astype(int) for p in A.PROTOS}; P = {}
for p in A.PROTOS:
    zs = [np.load(f) for f in sorted(glob.glob(f"{D}/oc_{arm}-{p}-s*.npz"))]; assert len(zs) == 3 and all((z["index"] == IDX[p]).all() for z in zs); P[p] = np.mean([z["proba"] for z in zs], 0)
A.EXTRA["ocu"] = P; REF = {p: Z[f"GE|{p}"].copy() for p in A.PROTOS}; LP = {p: np.log(np.clip(P[p], 1e-3, 1)) for p in A.PROTOS}
print(f"OCUNET ({arm}, mean of 3 seeds) alone: new {SC.person(P['all_loso'], 'all_loso').mean():.2f} seen {SC.person(P['seen_session'], 'seen_session').mean():.2f}")
for bn in env.EXPL:
    base = A.system(bn, A.FAMS)["out"]; print(SC.absline(bn + " chain", base), flush=True)
    for w in (0.5, 1.0, 2.0):
        print("   " + SC.line(f"mix member, weight {w}", SC.compare(A.system(bn, A.FAMS, extra=[("ocu", w)])["out"], base))[:118], flush=True)
    for beta in (0.05, 0.1, 0.25):
        out = {}
        for p in A.PROTOS:
            L = np.log(np.clip(base[p], 1e-9, 1)) + beta * LP[p]; e = np.exp(L - L.max(1, keepdims=True)); out[p] = e / e.sum(1, keepdims=True)
        print("   " + SC.line(f"final stage, beta {beta}", SC.compare(out, base))[:118], flush=True)
    for name, f in (("SessionEM view + net log posteriors (51)", lambda p: np.c_[REF[p], LP[p]]), ("SessionEM view = event rates + net log posteriors (19)", lambda p: np.c_[REF[p][:, 32:], LP[p]])):
        for p in A.PROTOS:
            Z[f"GE|{p}"] = f(p)
        print("   " + SC.line(name, SC.compare(A.system(bn, A.FAMS)["out"], base))[:118], flush=True)
        for p in A.PROTOS:
            Z[f"GE|{p}"] = REF[p]
