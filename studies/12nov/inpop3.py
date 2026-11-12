"""INPOP2 (ledger s96): ten-person proxy with personally fine-tuned nets (stored pfx outputs) and the v5.1 stage.
Usage: python studies/12nov/inpop3.py"""
import glob, os, sys
import numpy as np
F = os.path.abspath(__file__ + "/../inpop.py"); exec(compile(open(F).read().split("\nfor bn in (")[0], F, "exec"))  # noqa: S102  (defines chain, MEM, ba_people, PF, SLOTS)
PFV = {v: {a: np.mean([np.load(f)["proba"] for f in sorted(glob.glob(f"{D}/pfx_{a}_{v}-proxy-s*.npz"))], 0) for a in list(SLOTS) + ["sh"]} for v in ("pfh60", "pf60")}
BASE_PF = dict(PF)


def staged(P, beta=0.10):
    S = P.copy()
    for q in PEOPLE:
        m = MEM[q][1]; k = O.PER[B] == q; L = np.log(np.clip(P[k], 1e-9, 1)) + beta * sum(np.log(np.clip(m[x], 1e-3, 1)) for x in "TCS") / 3
        e = np.exp(L - L.max(1, keepdims=True)); S[k] = e / e.sum(1, keepdims=True)
    return S


for bn in ("B2_pool", "B2_s345", "B2_s012"):
    PF.clear(); PF.update(BASE_PF); P0 = chain(bn, True); r0 = ba_people(P0); st = ba_people(staged(P0)) - r0
    print(f"{bn}: in-population chain {r0.mean():.1f} | stage {st.mean():+.2f}", flush=True)
    for v in ("pfh60", "pf60"):
        PF.clear(); PF.update(PFV[v]); P1 = chain(bn, True); a = ba_people(P1) - r0; b = ba_people(staged(P1)) - r0; d = b - st
        bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
        print(f"   {v}: nets alone {a.mean():+.2f} | nets + stage {b.mean():+.2f} | against the stage alone {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] ahead {(d > 0).sum()}/10 | " + " ".join(f"{x:+.1f}" for x in d), flush=True)
    PF.clear(); PF.update(BASE_PF)
