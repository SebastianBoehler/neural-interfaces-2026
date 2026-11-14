"""s247b: package-form late fusion. Output = BASE x PFT^(w r) per window, r = the gate's ramp from the same chain run
(0 before the first routed window: those rows emit BASE exactly), PFT = geometric mean of the 15 re-streamed PFT nets
(pftshuf.py) of the session's person in that delivery order. BASE = ER or T of chain12.py (S2=0: protocol B, sessions 2 + 3
tested, = s245; S2=1: protocol A, session 3 only). Primary (pf200, w 0.25); cross-person: (variant, w) chosen on one
person, scored on the other. SAVE: per-window arms for score_ret.py (<order>|<base>|<arm>, seen_session rows).
Usage: PROT=B|A python studies/14nov/strongpersonal/fusion.py -> runs/strongpersonal/fusion_<PROT>.txt (+ fusion_<PROT>_out.npz; A: session-2 rows = s245 ER)"""
import glob, os
import numpy as np
import common as C

PROT = os.environ.get("PROT", "B"); SRC = f"runs/strongpersonal/chain12_s2{int(PROT == 'A')}.npz"
TESTS = ("2", "3") if PROT == "B" else ("3",); WS = (0.125, 0.25, 0.5, 1.0); VS = ("pf60", "pf200"); PRIM = ("pf200", 0.25)
Zc = np.load(f"{C.REPO}/{SRC}"); ORD = sorted({k.split("|")[0] for k in Zc.files}); BASES = sorted({k.split("|")[1] for k in Zc.files})
assert len(ORD) == 16 and len(BASES) == 6, (ORD, BASES)


def nets(v):
    fs = sorted(glob.glob(f"{C.OUT}/pftshuf/{PROT}_*_{v}-s*.npz")); assert len(fs) == 15, f"{PROT} {v}: {len(fs)} net files"
    out = {}
    for o in ORD + ["rec"]:
        L = []
        for f in fs:
            z = np.load(f); P = np.full((C.N, 3), 1 / 3); P[z["index"].astype(int)] = z[o]; L.append(np.log(np.clip(P, 1e-4, 1)))
        G = np.exp(np.mean(L, 0)); out[o] = (G / G.sum(1, keepdims=True))[C.SS]
    return out


PFT = {v: nets(v) for v in VS}
TE = {q: np.flatnonzero((C.PER[C.SS] == q) & np.isin(C.SES[C.SS], TESTS)) for q in C.SEEN}
def full(Pss):
    P = np.full((C.N, 3), 1 / 3); P[C.SS] = Pss; return P


def ba(q, Pss):
    return C.ba(C.SS[TE[q]], full(Pss))


def fuse(B, M, wr):
    L = np.log(np.clip(B, 1e-6, 1)) + wr[:, None] * np.log(np.clip(M, 1e-6, 1)); L -= L.max(1, keepdims=True); E = np.exp(L)
    return E / E.sum(1, keepdims=True)


lines = [f"PFT alone ({PROT}, sessions {TESTS}): " + "; ".join(f"{v} " + " ".join(f"{q} rec {ba(q, PFT[v]['rec']):.2f} shuf {np.mean([ba(q, PFT[v][o]) for o in ORD]):.2f}" for q in C.SEEN) for v in VS)]
R, SAVE = {}, {}  # R[base_arm, v, w][order] = (dP04, dP10) mean over bases
for o in ORD:
    for bn in BASES:
        ramp = Zc[f"{o}|{bn}|ramp"].astype(float); E0 = Zc[f"{o}|{bn}|ER"].astype(float); e0 = np.array([ba(q, E0) for q in C.SEEN])
        for arm in ("ER", "T"):
            B = Zc[f"{o}|{bn}|{arm}"].astype(float)
            R.setdefault((arm, None, 0.0), {}).setdefault(o, []).append(np.array([ba(q, B) for q in C.SEEN]) - e0)
            for v in VS:
                for w in WS:
                    F = fuse(B, PFT[v][o], w * ramp); R.setdefault((arm, v, w), {}).setdefault(o, []).append(np.array([ba(q, F) for q in C.SEEN]) - e0)
                    if (v, w) == PRIM:
                        SAVE[f"{o}|{bn}|{'F' if arm == 'ER' else 'TF'}"] = F.astype(np.float32)
            SAVE[f"{o}|{bn}|{arm}"] = B.astype(np.float32)
R = {k: {o: np.mean(x, 0) for o, x in d.items()} for k, d in R.items()}


def summ(k, who=(0, 1)):
    d = R[k]; sc = {s: np.mean([x for o, x in d.items() if o.startswith(s)], 0) for s in ("within", "full", "runs")}
    m = np.mean(list(sc.values()), 0); sw = 0.2 * m[0] + 0.8 * m[1]
    return m, sw, sc, sum(d[o][1] > 0 for o in ORD), min(0.2 * d[o][0] + 0.8 * d[o][1] for o in ORD)


lines.append("change against ER (mean of schemes; P04 / P10 / sealed 0.2 P04 + 0.8 P10); per scheme P04/P10; P10 > 0 orders; min sealed over orders")
for k in [("T", None, 0.0)] + [(a, v, w) for a in ("ER", "T") for v in VS for w in WS]:
    m, sw, sc, n10, mn = summ(k)
    lines.append(f"  {k[0]}{'' if k[1] is None else ' x ' + k[1] + '^' + str(k[2]) + 'r'}: {m[0]:+.2f} / {m[1]:+.2f} / {sw:+.2f} | "
                 + " ".join(f"{s} {x[0]:+.2f}/{x[1]:+.2f}" for s, x in sc.items()) + f" | P10>0 {n10}/16 | min {mn:+.2f}" + ("   <- primary" if (k[1], k[2]) == PRIM else ""))
lines.append("cross-person (variant, w chosen on one person by its own gain, scored on the other):")
for arm in ("ER", "T"):
    g = {(v, w): summ((arm, v, w))[0] for v in VS for w in WS}; base = summ((arm, None, 0.0))[0] if arm == "T" else np.zeros(2)
    k4 = max(g, key=lambda k: g[k][0]); k10 = max(g, key=lambda k: g[k][1])
    d10, d04 = g[k4][1] - base[1], g[k10][0] - base[0]
    lines.append(f"  on {arm}: chosen on P04 {k4} -> P10 {d10:+.2f}; chosen on P10 {k10} -> P04 {d04:+.2f}; sealed {0.2 * d04 + 0.8 * d10:+.2f} (increments over {arm})")
open(f"{C.OUT}/fusion_{PROT}.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
if PROT == "A":  # score_ret.py scores sessions 2 + 3: session 2 (leaky here) gets s245's ER in every arm, so changes = s3 change / 2
    ref = np.load(f"{C.REPO}/runs/chainout/s245_ret.npz"); s2 = C.SES[C.SS] == "2"
    for k in SAVE:
        o, bn, _ = k.split("|"); SAVE[k][s2] = ref[f"{o}|{bn}|ER"][s2]
np.savez_compressed(f"{C.OUT}/fusion_{PROT}_out.npz", **SAVE)
