import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
REPO = os.getcwd(); R4 = REPO + "/studies/11nov/rt4/route4.py"; ns = {"__file__": R4, "__name__": "route4"}
sys.argv = ["route4.py", REPO + "/runs/replay_data/bio_model_psd_d64.npz"]
exec(compile(open(R4).read().split("\nbases = sys.argv")[0], R4, "exec"), ns)
A, SC = ns["A"], ns["SC"]
pn = lambda o: (SC.person(o["all_loso"], "all_loso"), SC.person(o["seen_session"], "seen_session"))
rows = []
for bn in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    out = {}
    for lam in (0.15, 0.20, 0.25):
        A.LAM = lam; out["v3", lam] = pn(A.system(bn, A.FAMS)["out"]); out["v4", lam] = pn(ns["system4"](bn, "max", 10)[0])
    A.LAM = 0.15
    r = [(out[v, l][0] - out[v, 0.15][0]).mean() for v in ("v3", "v4") for l in (0.20, 0.25)] + [(out[v, l][1] - out[v, 0.15][1]).mean() for v in ("v3", "v4") for l in (0.20, 0.25)]
    rows.append(r); print(bn, "new d(lam .20/.25): v3 %+.2f %+.2f | v4.1 %+.2f %+.2f || seen: v3 %+.2f %+.2f | v4.1 %+.2f %+.2f" % tuple(r), flush=True)
m = np.mean(rows, 0); print("MEAN new: v3 %+.2f %+.2f | v4.1 %+.2f %+.2f || seen: v3 %+.2f %+.2f | v4.1 %+.2f %+.2f" % tuple(m))
