"""XSESS (ledger s37): own-space personal members, standalone across sessions and in the v4.1 chain replay.
Usage: python studies/12nov/xsess.py [BASE,BASE]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/studies/12nov"]
import ownspace as O  # noqa: E402

rows = lambda p, s: np.flatnonzero((O.PER == p) & (O.SES == s))  # noqa: E731
Q = {k: np.full((len(O.y), 3), np.nan) for k in "TCS"}
print("standalone, causal stream (cell-mean BA)\nperson train -> test     T     C     S   T+C  T+C+S")
for p in ("P04", "P10"):
    for a in "123":
        m = O.Personal([rows(p, a)])
        for b in "123":
            if a == b:
                continue
            r = rows(p, b); o = m.stream(r)
            print(f"{p}    s{a} -> s{b}     " + " ".join(f"{O.ba(r, v):5.1f}" for v in (o['T'], o['C'], o['S'], o['T'] + o['C'], o['T'] + o['C'] + o['S'])), flush=True)
            if a == "1":
                for k in "TCS":
                    Q[k][r] = o[k]
    r = np.concatenate([rows(p, "2"), rows(p, "3")])
    print(f"{p}    dev seen (s1 -> s2,s3), mean of the two sessions: T+C {np.mean([O.ba(rows(p, b), Q['T'][rows(p, b)] + Q['C'][rows(p, b)]) for b in '23']):.1f}")

# ---- chain: the v4.1 replay with the true person's personal-head posterior replaced
R4 = REPO + "/studies/11nov/rt4/route4.py"; ns = {"__file__": R4, "__name__": "route4"}
sys.argv = ["route4.py", REPO + "/runs/replay_data/bio_model_psd_d64.npz"]
exec(compile(open(R4).read().split("\nbases = sys.argv")[0], R4, "exec"), ns)
A, R3, SC = ns["A"], ns["R3"], ns["SC"]; NM = {}; _gm = R3.gate_model


def gate_model(proto, f, oof):
    out = _gm(proto, f, oof)
    for s, (Xs, ys, gi) in out[0].items():
        NM[(proto, s)] = (list(out[1]), np.asarray(gi))
    return out


R3.gate_model = gate_model
VAR = {"V1 (PS1+T+C)/3": lambda h, t, c, s: (h + t + c) / 3, "V2 (PS1+T+C+S)/4": lambda h, t, c, s: (h + t + c + s) / 4,
       "V3 (T+C)/2": lambda h, t, c, s: (t + c) / 2, "x  (PS1+T)/2": lambda h, t, c, s: (h + t) / 2, "x  (PS1+C)/2": lambda h, t, c, s: (h + c) / 2}
for bn in (os.environ.get("BASES", "B2_pool,B2_s345").split(",")):
    out0, _ = ns["system4"](bn, "max", 10); H0 = dict(A._H); print(SC.absline(bn + " v4.1", out0), flush=True)
    if bn == os.environ.get("BASES", "B2_pool,B2_s345").split(",")[0]:
        for (proto, s), (names, gi) in NM.items():
            if proto == "seen_session":
                j = [i for i, q in enumerate(names) if q.endswith(O.PER[gi[0]])]; assert len(j) == 1
                print(f"   {s}: PS1 head alone {O.ba(gi, H0[(proto, s)][j[0]]):.1f}")
    for tag, fn in VAR.items():
        for (proto, s), (names, gi) in NM.items():
            if proto == "seen_session":
                j = [i for i, q in enumerate(names) if q.endswith(O.PER[gi[0]])][0]; H = H0[(proto, s)].copy()
                H[j] = fn(H[j], Q["T"][gi], Q["C"][gi], Q["S"][gi]); assert not np.isnan(H[j]).any(); A._H[(proto, s)] = H
        out, _ = ns["system4"](bn, "max", 10); print("   " + SC.line(tag, SC.compare(out, out0)), flush=True)
        A._H.update(H0)
