"""MEMSWEEP (ledger s44): own-space member variants on the ten-person proxy (s41) and the 12 session pairs (s42).
Usage: python studies/12nov/memsweep.py"""
import os, sys, time
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/replay", REPO + "/studies/12nov"]
import env  # noqa: E402
import own2 as O  # noqa: E402
A = env.A
VAR = {"V0 current": {}, "V1 ov17": dict(bands=O.OV17), "V2 halves": dict(halves=True), "V3 csp 2+2": dict(csp_k=2), "V4 C shr 0.3": dict(shr_c=0.3),
       "V5 T C.003": dict(lr_c=0.003), "V6 T C.03": dict(lr_c=0.03), "V7 calibrated": dict(calib=True), "V8 ov17+calib": dict(bands=O.OV17, calib=True)}
BETAS = (0.1, 0.25, 0.5, 1.0); PEOPLE = sorted(set(O.PER)); lg = lambda P: np.log(np.clip(P, 1e-3, 1))  # noqa: E731
rows = lambda p, s: np.flatnonzero((O.PER == p) & (O.SES == s))  # noqa: E731
BASE = {bn: A.system(bn, A.FAMS)["out"]["all_loso"] for bn in ("B2_pool", "B2_s345")}
SPLIT = {}
for p in PEOPLE:
    s1 = rows(p, "1"); order = list(dict.fromkeys(O.RUN[s1])); SPLIT[p] = (s1[np.isin(O.RUN[s1], order[:3])], s1[np.isin(O.RUN[s1], order[3:])])
PAIRS = [(p, a, b) for p in ("P04", "P10") for a in "123" for b in "123" if a != b]


def stage(P, mem, beta):
    L = np.log(np.clip(P, 1e-9, 1)) + beta * sum(lg(mem[k]) for k in "TCS") / 3; e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


print("variant          | ten people: gain at beta " + " / ".join(map(str, BETAS)) + " (hurt > 1 at best) | 12 pairs: same | temps (median T C S)", flush=True)
for name, cfg in VAR.items():
    t0 = time.time(); M1, temps = {}, []
    for p in PEOPLE:
        m = O.Personal([SPLIT[p][0]], cfg); M1[p] = m.stream(SPLIT[p][1]); temps.append([m.temp[k] for k in "TCS"])
    M2 = {j: O.Personal([rows(j[0], j[1])], cfg).stream(rows(j[0], j[2])) for j in PAIRS}
    g1 = np.array([[np.mean([O.ba(SPLIT[p][1], stage(BASE[bn][SPLIT[p][1]], M1[p], b)) - O.ba(SPLIT[p][1], BASE[bn][SPLIT[p][1]]) for bn in BASE]) for p in PEOPLE] for b in BETAS])
    g2 = np.array([[np.mean([O.ba(rows(j[0], j[2]), stage(BASE[bn][rows(j[0], j[2])], M2[j], b)) - O.ba(rows(j[0], j[2]), BASE[bn][rows(j[0], j[2])]) for bn in BASE]) for j in PAIRS] for b in BETAS])
    i1, i2 = g1.mean(1).argmax(), g2.mean(1).argmax()
    print(f"{name:16s} | " + " / ".join(f"{v:+.2f}" for v in g1.mean(1)) + f" ({(g1[i1] < -1).sum()}) | " + " / ".join(f"{v:+.2f}" for v in g2.mean(1))
          + f" ({(g2[i2] < -1).sum()}) | " + " ".join(f"{v:.2f}" for v in np.median(temps, 0)) + f" | {time.time() - t0:.0f}s", flush=True)
    print("                   per person at best beta: " + " ".join(f"{p[1:]}:{v:+.1f}" for p, v in zip(PEOPLE, g1[i1])), flush=True)
