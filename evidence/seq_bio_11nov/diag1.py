"""Why does the Graz position prior hurt? Accuracy of chain stages / members on pattern vs off-pattern windows."""
import sys, collections
import numpy as np
ARCH = "/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026/runs/archive_scratch_11nov"
sys.path.insert(0, ARCH + "/secop")
import env
A = env.A; M = A.M; SC = A.SC; L = A.L
z = np.load(env.REPO + "/data/proteus_windows.npz", allow_pickle=True)
y = z["y"]; per = np.array([s[-3:] for s in z["subject"]]); ses = z["session"]; task = z["task"].astype(str); run = z["run"]
N = len(y); pos = np.zeros(N, int)
for k in set(zip(per, ses, run)):
    m = np.flatnonzero((per == k[0]) & (ses == k[1]) & (run == k[2])); pos[m] = np.arange(len(m))
graz = np.char.endswith(task, "Graz"); pat = np.array([2, 1, 0])[pos % 3]; match = (y == pat)
p = "all_loso"; bn = "B2_pool"
print("prior-only acc on Graz %.3f  BH %.3f" % (match[graz].mean(), match[~graz].mean()))
mem = A.members(bn, p); mixv, _ = A.mix(bn, p, A.FAMS)
b = M.rebal(mixv, p, None); g = A.gm.gate(b, p); e = A.gm.em_g(b, p, g); t = A.gm.tsem_g(e, p, g)
def rep(name, P):
    pr = P.argmax(1); ok = pr == y
    s = f"{name:10s} Graz acc {ok[graz].mean():.3f} | on-pattern {ok[graz & match].mean():.3f} off {ok[graz & ~match].mean():.3f} | pred==pattern rate {np.mean(pr[graz] == pat[graz]):.3f}"
    s += f" || BH acc {ok[~graz].mean():.3f} on {ok[~graz & match].mean():.3f} off {ok[~graz & ~match].mean():.3f} pred==pat {np.mean(pr[~graz] == pat[~graz]):.3f}"
    print(s)
for k, v in mem.items(): rep(str(k), v)
for name, P in (("mix", mixv), ("rebal", b), ("em", e), ("tsem", t)): rep(name, P)
# by previous-label relation: is the truth equal to label of previous window? two back?
prev = np.r_[-1, y[:-1]]; same_run = np.r_[False, (per[1:] == per[:-1]) & (ses[1:] == ses[:-1]) & (run[1:] == run[:-1])]
for name, P in (("mix", mixv), ("tsem", t), ("c1", mem["c1"]), ("m6", mem[6])):
    pr = P.argmax(1); ok = pr == y
    for nm, sel in (("Graz", graz), ("BH", ~graz)):
        a = sel & same_run
        print(f"{name:5s} {nm:4s} acc when y==prev {ok[a & (y == prev)].mean():.3f} (n {int((a & (y == prev)).sum())}) | y!=prev {ok[a & (y != prev)].mean():.3f} | P(pred==prev) {np.mean(pr[a] == prev[a]):.3f} vs P(y==prev) {np.mean(y[a] == prev[a]):.3f}")
