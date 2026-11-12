"""V5R (ledger s45): v4.1 chain replay + the own-space stage where the real merged gate routes, at the gate's ramp,
with the gallery person's members. Usage: [BASES=..] [BETA=0.10] python studies/12nov/v5_replay.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path[:0] = [REPO + "/studies/12nov"]
import ownspace as O  # noqa: E402
R4 = REPO + "/studies/11nov/rt4/route4.py"; ns = {"__file__": R4, "__name__": "route4"}
sys.argv = ["route4.py", REPO + "/runs/replay_data/bio_model_psd_d64.npz"]
exec(compile(open(R4).read().split("\nbases = sys.argv")[0], R4, "exec"), ns)
A, R3, SC, RL = ns["A"], ns["R3"], ns["SC"], ns["RL"]; BETA = float(os.environ.get("BETA", 0.10))
NM, REC = {}, {}; _gm, _sm = R3.gate_model, R3.slot_mix


def gate_model(proto, f, oof):
    out = _gm(proto, f, oof)
    for s, (Xs, ys, gi) in out[0].items():
        NM[(proto, s)] = (list(out[1]), np.asarray(gi), f)
    return out


def slot_mix(mix, c1, H, r, j):
    key = next(k for k, v in A._H.items() if v is H); REC[key] = (np.asarray(r, float).copy(), np.asarray(j).copy()); return _sm(mix, c1, H, r, j)


R3.gate_model, R3.slot_mix = gate_model, slot_mix
MODELS, STREAMS = {}, {}


def members(proto, s, q):
    names, gi, f = NM[(proto, s)]
    if (proto, s, q) not in STREAMS:
        if (proto, f, q) not in MODELS:
            itr = np.load(f"{RL.FE}/{proto}-{f}.npz")["idx_tr"]; itr = itr[O.PER[itr] == q[-3:]]
            MODELS[(proto, f, q)] = O.Personal([itr[O.SES[itr] == u] for u in sorted(set(O.SES[itr]))])
        STREAMS[(proto, s, q)] = MODELS[(proto, f, q)].stream(gi)
    return STREAMS[(proto, s, q)]


def staged(out0):
    out = {p: v.copy() for p, v in out0.items()}; touched = []
    for (proto, s), (r, j) in REC.items():
        names, gi, f = NM[(proto, s)]; idx = A.pl.B[f"idx|{proto}"]; pos = {int(g): k for k, g in enumerate(idx)}; rows = np.array([pos[int(g)] for g in gi])
        for t in np.flatnonzero(r > 0):
            m = members(proto, s, names[int(j[t])])
            L = np.log(np.clip(out0[proto][rows[t]], 1e-9, 1)) + BETA * r[t] * sum(np.log(np.clip(m[k][t], 1e-3, 1)) for k in "TCS") / 3
            e = np.exp(L - L.max()); out[proto][rows[t]] = e / e.sum()
        touched.append((proto, s.split("/")[-1], names[int(j[np.flatnonzero(r > 0)[0]])][-3:], int(np.flatnonzero(r > 0)[0]), float((r > 0).mean()), float(r[r > 0].mean())))
    return out, touched


res = []
for bn in os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(","):
    REC.clear(); out0, _ = ns["system4"](bn, "max", 10); out, touched = staged(out0); r = SC.compare(out, out0); res.append(r)
    print(f"{bn}: v4.1 J {0.5 * SC.person(out0['all_loso'], 'all_loso').mean() + 0.5 * SC.person(out0['seen_session'], 'seen_session').mean():.2f} | stage dJ {r['J']:+.2f} "
          f"new {r['new']:+.3f} seen {r['seen']:+.2f} (P04 {r['ds'][0]:+.2f}, P10 {r['ds'][1]:+.2f})", flush=True)
    if len(res) == 1:
        for t in touched:
            print(f"      {t[0]:12s} {t[1]:7s} -> {t[2]} from window {t[3]}, routed share {t[4]:.2f}, mean ramp {t[5]:.2f}")
print(f"mean of {len(res)} bases: dJ {np.mean([r['J'] for r in res]):+.2f} new {np.mean([r['new'] for r in res]):+.3f} seen {np.mean([r['seen'] for r in res]):+.2f} "
      f"(P04 {np.mean([r['ds'][0] for r in res]):+.2f}, P10 {np.mean([r['ds'][1] for r in res]):+.2f})")
