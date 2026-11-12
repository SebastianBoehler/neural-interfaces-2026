"""TRUST2 (ledger s80): v5.1 replay with a trust-scaled beta (running agreement of the personal members with the chain).
Usage: python studies/12nov/trust2.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
F = REPO + "/studies/12nov/v5_replay.py"; exec(compile(open(F).read().split("\nres = []")[0], F, "exec"))  # noqa: S102
LO, HI, BLO, BHI, A0, K0 = 0.50, 0.70, 0.10, 0.40, 0.65, 10.0


def beta_path(m, P):
    mm = sum(np.log(np.clip(m[k], 1e-3, 1)) for k in "TCS"); ag = (mm.argmax(1) == P.argmax(1)).astype(float)
    run = (K0 * A0 + np.concatenate([[0.0], np.cumsum(ag)[:-1]])) / (K0 + np.arange(len(ag)))  # causal: windows before t
    return BHI + (BLO - BHI) * np.clip((run - LO) / (HI - LO), 0, 1)


def staged_trust(out0):
    out = {p: v.copy() for p, v in out0.items()}; bs = []
    for (proto, s), (r, j) in REC.items():
        names, gi, f = NM[(proto, s)]; idx = A.pl.B[f"idx|{proto}"]; pos = {int(g): k for k, g in enumerate(idx)}; rows = np.array([pos[int(g)] for g in gi])
        on = np.flatnonzero(r > 0)
        if not len(on):
            continue
        m = members(proto, s, names[int(j[on[0]])]); b = beta_path(m, out0[proto][rows])
        for t in on:
            mt = members(proto, s, names[int(j[t])])
            L = np.log(np.clip(out0[proto][rows[t]], 1e-9, 1)) + b[t] * r[t] * sum(np.log(np.clip(mt[k][t], 1e-3, 1)) for k in "TCS") / 3
            e = np.exp(L - L.max()); out[proto][rows[t]] = e / e.sum()
        bs.append((proto, s.split("/")[-1], names[int(j[on[0]])][-3:], float(b[on].mean())))
    return out, bs


res = []
for bn in ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2"):
    REC.clear(); out0, _ = ns["system4"](bn, "max", 10); fixed, _ = staged(out0); tr, bs = staged_trust(out0); r = SC.compare(tr, fixed); res.append(r)
    print(f"{bn}: trust against fixed 0.10: seen {r['seen']:+.2f} (P04 {r['ds'][0]:+.2f}, P10 {r['ds'][1]:+.2f}) new {r['new']:+.3f}", flush=True)
    if len(res) == 1:
        for b in bs:
            print(f"     {b[0]:12s} {b[1]:7s} as {b[2]}: mean beta {b[3]:.2f}")
ok = sum(r["seen"] >= 0.3 and r["ds"][1] >= -0.3 and r["new"] >= -0.05 for r in res)
print(f"MEAN: seen {np.mean([r['seen'] for r in res]):+.2f} (P04 {np.mean([r['ds'][0] for r in res]):+.2f}, P10 {np.mean([r['ds'][1] for r in res]):+.2f}) new {np.mean([r['new'] for r in res]):+.3f}; bases meeting the lever {ok}/6")
