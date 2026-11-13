"""TORACLE (ledger s170): the time-weighted ORACLE stages (timeoracle.py) on v5.1b as uploaded, new and returning people
(shufpkg.system: routing v4.1 + own-space stage, gate calibration fixed). H = 30 and 90 windows; H = inf must reproduce
the shipped stages (asserted, recording order). Orders: recording, full, runs, within (PERMS each).
Usage: [SCHEMES=..] [PERMS=1] [BASES=..] python studies/13nov/toracle.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PERMS", "1")
import shufpkg as K  # noqa: E402  (no run on import)
import timeoracle as TO  # noqa: E402
S, PR, BASES = K.S, K.PR, K.BASES; HS = [float(h) for h in os.environ.get("HS", "30,90").split(",")]


def run(bn, P, oof, H):
    if H is None:
        TO.uninstall()
    else:
        TO.install({S.N[p]: P[p].astype(float) for p in PR}, H)  # rows are in delivery order; P[p] holds each row's recording position
    try:
        return K.system(bn, P, oof)[:2]
    finally:
        TO.uninstall()


ident = {p: np.arange(S.N[p]) for p in PR}; REF, OOF, R = {}, {}, {}
for bn in BASES:
    REF[bn], OOF[bn] = run(bn, ident, None, None)
    chk, _ = run(bn, ident, OOF[bn], np.inf)
    assert all(np.allclose(chk[p], REF[bn][p]) for p in PR), f"{bn}: H = inf does not reproduce the shipped stages"
print("check passed: H = inf reproduces the shipped stages in recording order", flush=True)
ORD = [("rec", 0)] + [(sc, sd) for sc in S.SCHEMES for sd in range(S.PERMS)]
for sc, sd in ORD:
    P = ident if sc == "rec" else S.perm_for(sc, sd)
    for bn in BASES:
        for H in [None] + HS:
            R[sc, sd, bn, H] = run(bn, P, OOF[bn], H)[0]
    print(f"  {sc} {sd} done", flush=True)
S.apply(ident)
print("against v5.1b in recording order, six-base mean: new | seen (P04, P10) | J; ORACLE rows weight delivered windows by true recording distance")
for sc in ["rec"] + S.SCHEMES:
    keys = [k for k in R if k[0] == sc]
    for H in [None] + HS:
        dn = np.mean([R[k][p0] - REF[k[2]][p0] for k in keys if k[3] == H for p0 in ["all_loso"]], 0)
        ds = np.mean([R[k]["seen_session"] - REF[k[2]]["seen_session"] for k in keys if k[3] == H], 0)
        print(f"  {sc:7s} {'shipped' if H is None else f'oracle H {H:.0f}':12s} new {dn.mean():+.2f} | seen {ds.mean():+.2f} (P04 {ds[0]:+.2f}, P10 {ds[1]:+.2f}) | J {0.5 * dn.mean() + 0.5 * ds.mean():+.2f}", flush=True)
np.savez(S.REPO + "/runs/toracle.npz", **{f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|{p}": v[p] for k, v in R.items() for p in PR})
