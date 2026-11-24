import sys, os, numpy as np
sys.dont_write_bytecode = True
REPO = "<repo>"
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; L = A.L
# 1. which member slots differ across bases
for p in A.PROTOS:
    ref = L.Z[f"B2_pool|Ps|{p}"]
    print(p, "Ps shape", ref.shape)
    for bn in env.BASES[1:]:
        Ps = L.Z[f"{bn}|Ps|{p}"]
        print("  ", bn, "slots differing from B2_pool:", [(i, float(np.abs(Ps[i]-ref[i]).max().round(3))) for i in range(len(Ps)) if np.abs(Ps[i]-ref[i]).max() > 1e-9])
# 2. chain_final origin
CH = np.load(REPO + "/studies/12nov/chain_final.npz")
print({k: CH[k].shape for k in CH.files})
for bn in env.BASES:
    r = A.system(bn, A.FAMS)["out"]
    print(bn, "max |chain_final - system out| all_loso %.3g seen %.3g" % (np.abs(r["all_loso"] - CH["all_loso"]).max(), np.abs(r["seen_session"] - CH["seen_session"]).max()))
print("idx_seen equal:", (CH["idx_seen"] == A.pl.B["idx|seen_session"]).all())
y = A.SC.B["y|all_loso"]; print("label acc all_loso", np.mean(CH["all_loso"].argmax(1) == y), "mean conf", CH["all_loso"].max(1).mean())
