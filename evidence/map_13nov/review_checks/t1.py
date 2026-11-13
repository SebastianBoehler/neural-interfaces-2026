import sys, glob, os, time
import numpy as np
REPO = os.getcwd()
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC
CH = np.load(REPO + "/studies/12nov/chain_final.npz")
print("chain_final keys", {k: CH[k].shape for k in CH.files})
print("idx_seen equal:", (CH["idx_seen"] == A.pl.B["idx|seen_session"]).all())
y = {p: SC.B[f"y|{p}"] for p in A.PROTOS}
for p in A.PROTOS:
    print(p, "chain_final label acc", np.mean(CH[p].argmax(1) == y[p]))
t0 = time.time(); r = A.system("B2_pool", A.FAMS); print("A.system time", time.time() - t0)
for p in A.PROTOS:
    print("vs A.system(B2_pool) routed out", p, np.abs(CH[p] - r["out"][p]).max())
for f in sorted(glob.glob(REPO + "/runs/ttt/chain*.npz")):
    z = np.load(f)
    try:
        print(os.path.basename(f), {p: float(np.abs(CH[p] - z[p]).max()) for p in A.PROTOS}, "acc", {p: round(float(np.mean(z[p].argmax(1) == y[p])), 4) for p in A.PROTOS})
    except Exception as e:
        print(os.path.basename(f), "err", e, z.files)
# causality check of chain_final: is row t of the routed chain a function of rows <= t? truncation test on one session
