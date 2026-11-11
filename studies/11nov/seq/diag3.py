"""Flip anatomy of the Graz position prior at each chain stage, and calibration by on/off-pattern."""
import sys, collections
import numpy as np
REPO = __import__("os").path.abspath(__file__ + "/../../../..")
sys.path.insert(0, REPO + "/replay")
import env
A = env.A; M = A.M; SC = A.SC
z = np.load(env.REPO + "/data/proteus_windows.npz", allow_pickle=True)
y = z["y"]; per = np.array([s[-3:] for s in z["subject"]]); ses = z["session"]; task = z["task"].astype(str); run = z["run"]
N = len(y); pos = np.zeros(N, int)
for k in set(zip(per, ses, run)):
    m = np.flatnonzero((per == k[0]) & (ses == k[1]) & (run == k[2])); pos[m] = np.arange(len(m))
graz = np.char.endswith(task, "Graz"); pm = pos % 3; pat = np.array([2, 1, 0])[pm]; on = y == pat
T = np.full((3, 3), 0.22); T[0, 2] = T[1, 1] = T[2, 0] = 0.56
LP = np.where(graz[:, None], np.log(3 * T[pm]), 0.0)
p = "all_loso"; bn = "B2_pool"
mixv, _ = A.mix(bn, p, A.FAMS); b = M.rebal(mixv, p, None); g = A.gm.gate(b, p); e = A.gm.em_g(b, p, g); t = A.gm.tsem_g(e, p, g)
G = graz
for name, P, a in (("mix", mixv, 1.76), ("rebal", b, 2.33), ("em", e, 2.1), ("tsem", t, 1.92)):
    lP = a * np.log(np.clip(P, 1e-9, 1)); pc = np.exp(lP - lP.max(1, keepdims=True)); pc /= pc.sum(1, keepdims=True)
    q = np.exp(lP + LP - (lP + LP).max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    p0, p1 = P.argmax(1), q.argmax(1); fl = G & (p0 != p1)
    print(f"\n{name} (temp {a}): Graz n {G.sum()}, acc {np.mean(p0[G] == y[G]):.3f} -> {np.mean(p1[G] == y[G]):.3f}; model-expected acc {pc[G].max(1).mean():.3f} -> {q[G].max(1).mean():.3f}")
    print(f"   flips {fl.sum()} ({fl.sum() / G.sum():.1%}): fixed {np.sum(fl & (p1 == y))}, broken {np.sum(fl & (p0 == y))}, wrong->wrong {np.sum(fl & (p0 != y) & (p1 != y))}; flips to pattern class {np.mean(p1[fl] == pat[fl]):.2f}")
    # reliability of calibrated posterior by on/off pattern: P(top correct) vs mean top prob
    for nm, sel in (("on-pattern ", G & on), ("off-pattern", G & ~on)):
        print(f"   {nm}: n {sel.sum()} acc {np.mean(p0[sel] == y[sel]):.3f} mean cal top {pc[sel].max(1).mean():.3f} | cal prob of true class {pc[sel, y[sel]].mean():.3f} | pred==pat {np.mean(p0[sel] == pat[sel]):.3f}")
    # where chain top != pattern class and low margin: who is right?
    s = np.sort(pc, 1); lowm = G & (s[:, -1] / s[:, -2] < 2.5) & (p0 != pat)
    sec = np.argsort(pc, 1)[:, -2]
    k2 = lowm & (sec == pat)
    print(f"   low-odds windows with runner-up == pattern class: n {k2.sum()}; truth = top {np.mean(y[k2] == p0[k2]):.3f}, truth = pattern/runner-up {np.mean(y[k2] == pat[k2]):.3f}")
    k3 = G & (p0 != pat)
    print(f"   all windows with top != pattern: n {k3.sum()}; truth = top {np.mean(y[k3] == p0[k3]):.3f}; truth = pattern {np.mean(y[k3] == pat[k3]):.3f}  | top == pattern: n {np.sum(G & (p0 == pat))}, truth = top {np.mean(y[G & (p0 == pat)] == pat[G & (p0 == pat)]):.3f}")
