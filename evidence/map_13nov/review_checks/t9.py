import os, sys, warnings, glob
import numpy as np
warnings.filterwarnings("ignore")
REPO = os.getcwd(); R4 = REPO + "/studies/11nov/rt4/route4.py"; ns = {"__file__": R4, "__name__": "route4"}
sys.argv = ["route4.py", REPO + "/runs/replay_data/bio_model_psd_d64.npz"]
exec(compile(open(R4).read().split("\nbases = sys.argv")[0], R4, "exec"), ns)
A, SC = ns["A"], ns["SC"]; Z = A.L.Z
IDX = {p: Z[f"idx|{p}"].astype(int) for p in A.PROTOS}; P = {}
for p in A.PROTOS:
    zs = [np.load(f"{REPO}/runs/ocunet/probs/oc_nomix-{p}-s{s}.npz") for s in (0, 1, 2)]; assert all((z["index"] == IDX[p]).all() for z in zs); P[p] = np.mean([z["proba"] for z in zs], 0)
REF = {p: Z[f"GE|{p}"].copy() for p in A.PROTOS}; GEO = {p: np.c_[REF[p], np.log(np.clip(P[p], 1e-3, 1))] for p in A.PROTOS}
pn = lambda o: (SC.person(o["all_loso"], "all_loso"), SC.person(o["seen_session"], "seen_session"))
rows = []
for bn in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    b3 = pn(A.system(bn, A.FAMS)["out"]); b4 = pn(ns["system4"](bn, "max", 10)[0])
    for p in A.PROTOS:
        Z[f"GE|{p}"] = GEO[p]
    try:
        o3 = pn(A.system(bn, A.FAMS)["out"]); o4 = pn(ns["system4"](bn, "max", 10)[0])
    finally:
        for p in A.PROTOS:
            Z[f"GE|{p}"] = REF[p]
    r = [(o3[0] - b3[0]).mean(), (o3[1] - b3[1]).mean(), (o3[1] - b3[1])[0], (o3[1] - b3[1])[1], (o4[0] - b4[0]).mean(), (o4[1] - b4[1]).mean(), (o4[1] - b4[1])[0], (o4[1] - b4[1])[1]]
    rows.append(r); print(bn, "OCU view: v3 new %+.2f seen %+.2f (P04 %+.2f P10 %+.2f) | v4.1 new %+.2f seen %+.2f (P04 %+.2f P10 %+.2f)" % tuple(r), flush=True)
m = np.mean(rows, 0); print("MEAN v3 new %+.2f seen %+.2f (P04 %+.2f P10 %+.2f) dJ %+.2f | v4.1 new %+.2f seen %+.2f (P04 %+.2f P10 %+.2f) dJ %+.2f" % (m[0], m[1], m[2], m[3], 0.5 * (m[0] + m[1]), m[4], m[5], m[6], m[7], 0.5 * (m[4] + m[5])))
