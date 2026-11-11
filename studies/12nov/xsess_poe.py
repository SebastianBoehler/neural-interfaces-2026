"""XSESS follow-up (exploratory, ledger s37): own-space personal members as a tempered product instead of diluting PS1.
F: final stage on the chain output of the seen sessions; H: inside the personal head's slot (PS1 x members^beta)."""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/studies/12nov"]
import ownspace as O  # noqa: E402
rows = lambda p, s: np.flatnonzero((O.PER == p) & (O.SES == s))  # noqa: E731
Q = {k: np.full((len(O.y), 3), np.nan) for k in "TCS"}
for p in ("P04", "P10"):
    m = O.Personal([rows(p, "1")])
    for b in "23":
        o = m.stream(rows(p, b))
        for k in "TCS":
            Q[k][rows(p, b)] = o[k]
R4 = REPO + "/studies/11nov/rt4/route4.py"; ns = {"__file__": R4, "__name__": "route4"}
sys.argv = ["route4.py", REPO + "/runs/replay_data/bio_model_psd_d64.npz"]
exec(compile(open(R4).read().split("\nbases = sys.argv")[0], R4, "exec"), ns)
A, R3, SC, pl = ns["A"], ns["R3"], ns["SC"], ns["pl"]; NM = {}; _gm = R3.gate_model


def gate_model(proto, f, oof):
    out = _gm(proto, f, oof)
    for s, (Xs, ys, gi) in out[0].items():
        NM[(proto, s)] = (list(out[1]), np.asarray(gi))
    return out


R3.gate_model = gate_model
lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
EXP = {"TC": lambda g: (lg(Q["T"][g]) + lg(Q["C"][g])) / 2, "TCS": lambda g: (lg(Q["T"][g]) + lg(Q["C"][g]) + lg(Q["S"][g])) / 3,
       "C": lambda g: lg(Q["C"][g]), "S": lambda g: lg(Q["S"][g])}
nz = lambda L: np.exp(L - L.max(1, keepdims=True)) / np.exp(L - L.max(1, keepdims=True)).sum(1, keepdims=True)  # noqa: E731
for bn in os.environ.get("BASES", "B2_pool,B2_s345").split(","):
    out0, _ = ns["system4"](bn, "max", 10); H0 = dict(A._H); print(SC.absline(bn + " v4.1", out0), flush=True)
    idx = pl.B["idx|seen_session"]; pos = {int(g): k for k, g in enumerate(idx)}
    for name, fn in EXP.items():
        for beta in (0.25, 0.5, 1.0):
            out = {p: v.copy() for p, v in out0.items()}
            for (proto, s), (names, gi) in NM.items():
                if proto == "seen_session":
                    r = np.array([pos[int(g)] for g in gi]); out[proto][r] = nz(np.log(np.clip(out0[proto][r], 1e-9, 1)) + beta * fn(gi))
            print("   " + SC.line(f"F {name} beta {beta}", SC.compare(out, out0)), flush=True)
        for beta in (0.5, 1.0):
            for (proto, s), (names, gi) in NM.items():
                if proto == "seen_session":
                    j = [i for i, q in enumerate(names) if q.endswith(O.PER[gi[0]])][0]; H = H0[(proto, s)].copy()
                    H[j] = nz(np.log(np.clip(H[j], 1e-9, 1)) + beta * fn(gi)); A._H[(proto, s)] = H
            out, _ = ns["system4"](bn, "max", 10); print("   " + SC.line(f"H {name} beta {beta}", SC.compare(out, out0)), flush=True)
            A._H.update(H0)
