"""PSYS on the real later sessions (ledger s112 follow-up): the personal system fitted on session 1 of P04 / P10, streamed
over sessions 2-3, fused next to the v5.1 stage in the v5.1 replay (seen protocol), six bases.
Usage: python studies/12nov/psys_seen.py [smoke]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); ARGV = list(sys.argv)  # v5_replay's head overwrites sys.argv
F = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"))  # noqa: S102
G2 = {}; src = open(REPO + "/studies/12nov/psys.py").read(); exec(compile(src[src.index("def personal_system"):src.index("\n\nSYS = {}")], "psys", "exec"), G2)  # noqa: S102
G2.update(O=O, np=np, PCA=__import__("sklearn.decomposition", fromlist=["PCA"]).PCA)
SMOKE = "smoke" in ARGV; lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
SYSP = {}
for q in ("P04", "P10"):
    a = np.flatnonzero((O.PER == q) & (O.SES == "1"))
    for sn in "23":
        r = np.flatnonzero((O.PER == q) & (O.SES == sn)); SYSP[int(r[0])] = (r, G2["personal_system"](a, r))
res = []
for bn in ("B2_pool",) if SMOKE else ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    REC.clear(); out0, _ = ns["system4"](bn, "max", 10); v51, _ = staged(out0); out = {k: v.copy() for k, v in v51.items()}
    idx = A.pl.B["idx|seen_session"]; pos = {int(g): k for k, g in enumerate(idx)}
    for (proto, s), (r_, j) in REC.items():
        names, gi, f = NM[(proto, s)]; on = np.flatnonzero(r_ > 0)
        if proto != "seen_session" or not len(on) or int(gi[0]) not in SYSP:
            continue
        rows, Ps = np.array([pos[int(g)] for g in gi]), SYSP[int(gi[0])][1]
        for t in on:
            L = np.log(np.clip(v51[proto][rows[t]], 1e-9, 1)) + 0.25 * r_[t] * lg(Ps[t]); e = np.exp(L - L.max()); out[proto][rows[t]] = e / e.sum()
    r = SC.compare(out, v51); res.append(r); print(f"{bn}: personal system (0.25) on top of v5.1: seen {r['seen']:+.2f} (P04 {r['ds'][0]:+.2f}, P10 {r['ds'][1]:+.2f}) new {r['new']:+.3f}", flush=True)
print(f"MEAN: seen {np.mean([r['seen'] for r in res]):+.2f} (P04 {np.mean([r['ds'][0] for r in res]):+.2f}, P10 {np.mean([r['ds'][1] for r in res]):+.2f})")
