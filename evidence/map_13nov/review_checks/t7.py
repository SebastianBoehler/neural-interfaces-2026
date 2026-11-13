import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
REPO = os.getcwd(); F = REPO + "/studies/12nov/segfp3.py"
src = open(F).read().split("\nJf = lambda")[0]
g = {"__file__": F, "__name__": "segfp3"}; exec(compile(src, F, "exec"), g)
A, SC, ns, Z = g["A"], g["SC"], g["ns"], g["Z"]
Jf = lambda o: (SC.person(o["all_loso"], "all_loso").mean(), SC.person(o["seen_session"], "seen_session").mean())
gm_orig = g["_gm"]
def gm_aligned(proto, f, oof):
    G = np.full((2655, 3), np.nan); G[Z["idx|all_loso"].astype(int)] = oof; return gm_orig(proto, f, G)
bn = "B2_pool"
for kind in ("session", "order Graz,BH", "reset itf"):
    res = {}
    for tag, fn in (("as coded", gm_orig), ("oof aligned", gm_aligned)):
        g["_gm"] = fn; g["arrange"](kind)
        if tag == "as coded":
            print(kind, "| idx|all_loso == arange after arrange:", bool((Z["idx|all_loso"] == np.arange(2655)).all()))
        v3, i3 = ns["system4"](bn, "v3", 10); v4, i4 = ns["system4"](bn, "max", 10)
        res[tag] = (Jf(v3), Jf(v4), len([i for i in i3 if i[0] == "all_loso"]), len([i for i in i3 if i[0] == "seen_session"]), len([i for i in i4 if i[0] == "all_loso"]), len([i for i in i4 if i[0] == "seen_session"]))
        r = res[tag]; print(f"   {tag:12s}: v3 new {r[0][0]:.3f} seen {r[0][1]:.3f} J {0.5*sum(r[0]):.3f} | v4.1 new {r[1][0]:.3f} seen {r[1][1]:.3f} J {0.5*sum(r[1]):.3f} | routed streams v3 new/seen {r[2]}/{r[3]}, v4.1 {r[4]}/{r[5]}", flush=True)
g["_gm"] = gm_orig; g["arrange"]("session")
