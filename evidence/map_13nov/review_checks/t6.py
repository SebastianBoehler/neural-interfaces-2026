import sys, os, glob
import numpy as np
REPO = os.getcwd(); sys.path.insert(0, REPO + "/replay")
import env
A = env.A; SC = A.SC; R3 = A.R3
_mix = A.mix; _sm = R3.slot_mix
def mix09(*a, **k):
    m, s = _mix(*a, **k); q = m ** 0.9; return q / q.sum(1, keepdims=True), s
def sm_consistent(mix, c1, H, r, j):  # slot substitution on the un-powered mix, then the power
    m = mix ** (1 / 0.9); m = m / m.sum(1, keepdims=True); o = _sm(m, c1, H, r, j); q = o ** 0.9; return q / q.sum(1, keepdims=True)
for bn in ("B2_pool", "B2_s345", "B2_s0"):
    base = A.system(bn, A.FAMS); b = (SC.person(base["out"]["all_loso"], "all_loso"), SC.person(base["out"]["seen_session"], "seen_session"))
    A.mix = mix09; a1 = A.system(bn, A.FAMS); R3.slot_mix = sm_consistent; a2 = A.system(bn, A.FAMS); R3.slot_mix = _sm; A.mix = _mix
    f = lambda o: (SC.person(o["out"]["all_loso"], "all_loso") - b[0], SC.person(o["out"]["seen_session"], "seen_session") - b[1])
    d1, d2 = f(a1), f(a2)
    print(f"{bn}: MIX as coded: new {d1[0].mean():+.3f} seen {d1[1].mean():+.3f} (P04 {d1[1][0]:+.2f}, P10 {d1[1][1]:+.2f}) routed new/seen {a1['nnew']}/{a1['nseen']} | consistent slot substitution: new {d2[0].mean():+.3f} seen {d2[1].mean():+.3f} (P04 {d2[1][0]:+.2f}, P10 {d2[1][1]:+.2f})")
# inpop proxy rows
D = REPO + "/runs/ttt/probs_pfx"
fs = glob.glob(f"{D}/pfx_fb_base-proxy-s0.npz")
if fs:
    w = np.load(REPO + "/data/proteus_windows.npz", allow_pickle=True); per = np.array([s[-3:] for s in w["subject"]]); ses = w["session"].astype(str); run = w["run"].astype(int)
    idx = np.load(fs[0])["index"]; import collections
    print("inpop proxy rows: sessions", sorted(collections.Counter(zip(per[idx], ses[idx])).items())[:14], "runs", sorted(set(run[idx])))
