import sys, os
import numpy as np
REPO = os.getcwd(); sys.path.insert(0, REPO + "/replay")
import env, grid as G
A = env.A; SC = A.SC
PCC = np.load(REPO + "/runs/replay_data/post_cache.npz")
for bn in env.BASES:
    g = G.scores(PCC[f"{bn}|tsem|all_loso"], "all_loso"); s = A.system(bn, A.FAMS, routing=False)["out"]["all_loso"]; r = A.system(bn, A.FAMS)
    print(bn, "grid chain new %.2f | gated-boost chain, no routing new %.2f | routed v3 new %.2f (routed new sessions %d) | label agreement grid vs gated %.4f" % (g.mean(), SC.person(s, "all_loso").mean(), SC.person(r["out"]["all_loso"], "all_loso").mean(), r["nnew"], np.mean(PCC[f"{bn}|tsem|all_loso"].argmax(1) == s.argmax(1))))
