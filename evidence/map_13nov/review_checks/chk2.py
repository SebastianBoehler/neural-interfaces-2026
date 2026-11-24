import sys, os, glob, numpy as np
sys.dont_write_bytecode = True
REPO = "<repo>"
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC
R = REPO + "/runs/ttt/"
ARCHS = ["fb", "fbl", "fbraw", "sinc"]
def ld(pfx, a, v, p):
    fs = sorted(glob.glob(f"{R}probs{pfx[-1]}/{pfx}_{a}_{v}-{p}-s*.npz")); return np.stack([np.load(f)["proba"] for f in fs])
sess = A.L.Z["sess|all_loso"]
pos = np.zeros(len(sess), int)
for s in np.unique(sess):
    r = np.flatnonzero(sess == s); pos[r] = np.arange(len(r))
print("session lengths", np.bincount(np.unique(sess, return_inverse=True)[1]))
# b) identity / no-op checks
for a in ARCHS:
    b5, b6, b7 = (ld(f"ttt{k}", a, "base", "all_loso") for k in (5, 6, 7))
    print(a, "base 5 vs 6 max %.3g, 6 vs 7 max %.3g; seeds differ (s0 vs s1) %.3g" % (np.abs(b5-b6).max(), np.abs(b6-b7).max(), np.abs(b6[0]-b6[1]).max()))
    f5, f6, f7 = ld("ttt5", a, "fixA", "all_loso"), ld("ttt6", a, "fixA", "all_loso"), ld("ttt7", a, "fixA", "all_loso")
    print("   fixA 5 vs 6 mean|d| %.4g max %.3g ; 6 vs 7 mean %.4g max %.3g; fixA-base(6) mean %.4g" % (np.abs(f5-f6).mean(), np.abs(f5-f6).max(), np.abs(f6-f7).mean(), np.abs(f6-f7).max(), np.abs(f6-b6).mean()))
    for v in ["cum10", "cons2", "soft", "step5", "head", "view2"]:
        x = ld("ttt6", a, v, "all_loso")
        print("   ttt6 %-6s vs base mean|d| %.4g | vs fixA mean|d| %.4g | windows<30 equal base: %s | first diff pos %d" % (v, np.abs(x-b6).mean(), np.abs(x-f6).mean(), bool((x[:, pos < 30] == b6[:, pos < 30]).all()), pos[np.abs(x-b6).max((0,2)) > 0].min()))
    for v in ["gA", "gE20", "gC10E20", "gC5", "gC10T20"]:
        x = ld("ttt7", a, v, "all_loso")
        print("   ttt7 %-7s vs base mean|d| %.4g | vs fixA mean|d| %.4g | first diff pos %d" % (v, np.abs(x-b7).mean(), np.abs(x-f7).mean(), pos[np.abs(x-b7).max((0,2)) > 0].min()))
    for v in ["fixW", "fixS", "fixPL", "fixT", "fix40", "fixLR"]:
        x = ld("ttt5", a, v, "all_loso")
        print("   ttt5 %-6s vs base mean|d| %.4g | vs fixA mean|d| %.4g" % (v, np.abs(x-b5).mean(), np.abs(x-f5).mean()))
