"""POPSTAGE (ledger s57c): own-space members fitted on the OTHER people, as a final stage for new-person sessions.
Usage: python studies/12nov/popstage.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import ownspace as O  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z; assert (Z["idx|all_loso"] == np.arange(len(O.y))).all()
PEOPLE = sorted(set(O.PER)); LM = np.zeros((len(O.y), 3)); acc = []
for p in PEOPLE:
    train = [np.flatnonzero((O.PER == q) & (O.SES == s)) for q in PEOPLE if q != p for s in sorted(set(O.SES[O.PER == q]))]
    m = O.Personal(train)
    for s in sorted(set(O.SES[O.PER == p])):
        r = np.flatnonzero((O.PER == p) & (O.SES == s)); o = m.stream(r); LM[r] = sum(np.log(np.clip(o[k], 1e-3, 1)) for k in "TCS") / 3
    r = np.flatnonzero(O.PER == p); acc.append(100 * np.mean(LM[r].argmax(1) == O.y[r])); print(f"  {p}: population T.C.S alone, window accuracy {acc[-1]:.1f}", flush=True)
print(f"population members alone: mean over people {np.mean(acc):.1f}")
for bn in env.EXPL:
    base = A.system(bn, A.FAMS)["out"]; print(SC.absline(bn + " chain", base), flush=True)
    for beta in (0.05, 0.10, 0.25):
        out = {k: v.copy() for k, v in base.items()}; L = np.log(np.clip(base["all_loso"], 1e-9, 1)) + beta * LM; e = np.exp(L - L.max(1, keepdims=True)); out["all_loso"] = e / e.sum(1, keepdims=True)
        r = SC.compare(out, base); print(f"   beta {beta}: new {r['new']:+.2f} (G1 {np.mean([r['dn'][SC.PEOPLE.index(q)] for q in SC.G1]):+.2f}, G2 {np.mean([r['dn'][SC.PEOPLE.index(q)] for q in SC.G2]):+.2f}; "
                                 f"{r['npos']}/10 people >= 0) | " + " ".join(f"{q[1:]}:{v:+.1f}" for q, v in zip(SC.PEOPLE, r["dn"])), flush=True)
