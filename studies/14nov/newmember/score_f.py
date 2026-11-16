"""NEWMEMBER scoring (ledger s289: stages refit, arm A pool nested by person; same scoring as score_w.py), new people. Per order and base the V6Fr outputs of the control and of each arm
(runs/chainout/nm_<NAME>_<order>.npz from studies/14nov/newmember/run.py), raw and through the detector-gated count filter
(studies/14nov/stack/score.py's outs(): latch + J-wprior urn, emitted output only). Cell = per-person change arm - ctl (both
run through this wrapper, same filter), mean over bases. 80 % intervals resample people and orders (orders within scheme, schemes
equal weight); 'people up' = person means over orders; per-scheme groups are printed. A zero-weight arm Z is asserted bit-equal to ctl.
Returning people (seen_session) are not covered: the new-family files exist for all_loso only.
Usage: ARMS=nmAeegn,nmBeegnconf python studies/14nov/newmember/score.py ORDER [ORDER ...]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); sys.argv, ORD = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, REPO + "/studies/14nov/rundetect")
import package as K  # noqa: E402
ARMS = [f"nmA_w{w}" for w in "0.1 0.2 0.3 0.5".split()] + ["nmA_w1.0"] + [f"nmA_w0.5_r{f}x{x}" for f in "core1 hilb shallow teacher sc_fb sc_fbl sc_sinc".split() for x in ("0.5", "1.5")]; SCHEMES = ("within", "full", "runs")


def outs(f, order, hc, lat):
    z = np.load(f); bases = sorted({k.split("|")[0] for k in z.files if k.endswith("|V6Fr")}); raw, fil = [], []
    for bn in bases:
        P = np.full((K.C.N, 3), 1 / 3); P[K.ROWS] = z[f"{bn}|V6Fr"].astype(np.float64)
        raw.append(K.C.A.SC.person(P[K.ROWS], K.PROTO)); fil.append(K.C.A.SC.person(K.gate(P, K.J.filt_joint(P, hc, "urn"), order, lat)[K.ROWS], K.PROTO))
    return bases, np.mean(raw, 0), np.mean(fil, 0)


def path(name, order):
    return f"{REPO}/runs/chainout/nm_{name}_{order}.npz"


R = {}
for order in ORD:
    lat = K.latch(order); hc = K.J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
    bc, c0, c1 = outs(path("ctl", order), order, hc, lat); R[order, "ctl+filter"] = c1 - c0; zc = np.load(path("ctl", order))
    for arm in ARMS:
        ba, a0, a1 = outs(path(arm, order), order, hc, lat); assert ba == bc, (arm, order, ba, bc)
        za = np.load(path(arm, order)); bits = [np.array_equal(za[k], zc[k]) for k in za.files if k.endswith("|V6Fr")]
        assert not any(bits), f"{arm} {order}: arm output equals the control for some base (the new slot has no effect)"
        R[order, arm] = a0 - c0; R[order, arm + "+filter"] = a1 - c0
    print(f"  {order:8s} latched {sum(w is not None for w in lat.values())}/{len(lat)} ctl raw {c0.mean():.2f} | " + " ".join(f"{k} {v.mean():+.2f}" for (o, k), v in R.items() if o == order), flush=True)

rng = np.random.default_rng(0); by = {s: [o for o in ORD if o.rstrip("0123456789") == s] for s in SCHEMES}; by = {s: v for s, v in by.items() if v}
OUT = []
def say(x=""):
    print(x); OUT.append(x)
def pm(arm, suf):  # (10,) person means: schemes equal weight, orders averaged; filtered = arm filtered minus ctl filtered... as score_w (arm+filter vs ctl raw minus ctl filter effect)
    return np.mean([np.stack([R[o, arm + suf] - (R[o, "ctl+filter"] if suf else 0) for o in v]).mean(0) for v in by.values()], 0)
def sch(arm, suf):
    return {s: np.stack([R[o, arm + suf] - (R[o, "ctl+filter"] if suf else 0) for o in v]).mean() for s, v in by.items()}
def interval(fn):
    bt = [fn(rng.integers(0, 10, 10)) for _ in range(4000)]; return np.percentile(bt, [10, 90])
for suf, lab in (("", "raw"), ("+filter", "count-filtered")):
    PM = np.stack([pm(a, suf) for a in ARMS]); say(f"=== {lab}: arm A minus control (new people; schemes equal weight; people-resampled 80 % interval) ===")
    for j, a in enumerate(ARMS):
        lo, hi = interval(lambda pp, j=j: PM[j][pp].mean()); say(f"   {a:28s} {PM[j].mean():+.2f} [{lo:+.2f}, {hi:+.2f}] up {int((PM[j] > 0).sum())}/10 | " + " ".join(f"{s} {v:+.2f}" for s, v in sch(a, suf).items()))
    for pool_name, pool in (("all 19 arms", list(range(len(ARMS)))), ("refit pool only (w=1.0 and the 14 perturbations)", list(range(4, len(ARMS))))):
        P_ = PM[pool]
        sel = [pool[int(np.argmax(P_[:, [q for q in range(10) if q != p]].mean(1)))] for p in range(10)]; nested = np.array([PM[sel[p], p] for p in range(10)])
        def nested_fn(pp):
            v = []
            for i, p in enumerate(pp):
                rest = [r for r in range(len(pp)) if r != i]; v.append(PM[pool[int(np.argmax(P_[:, pp[rest]].mean(1)))], p])
            return np.mean(v)
        lo, hi = interval(nested_fn); bf = pool[int(np.argmax(P_.mean(1)))]
        ps = {s: np.mean([np.stack([R[o, ARMS[sel[p]] + suf][p] - (R[o, "ctl+filter"][p] if suf else 0) for o in v]).mean() for p in range(10)]) for s, v in by.items()}
        say(f"   NESTED [{pool_name}] {nested.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] people up {int((nested > 0).sum())}/10 | per scheme " + " ".join(f"{s} {v:+.2f}" for s, v in ps.items()) + f" | best fixed {ARMS[bf]} {PM[bf].mean():+.2f} (optimistic)")
        say("   selected per held-out person: " + " ".join(ARMS[x] for x in sel)); say("   nested people: " + " ".join(f"{x:+.2f}" for x in nested))
open(f"{REPO}/evidence/newmember_15nov/s289.txt", "w").write("\n".join(OUT) + "\n")
np.savez(f"{REPO}/runs/newmember/score_s289.npz", **{f"{o}|{k}": v for (o, k), v in R.items()})
