"""NEWMEMBER scoring (ledger s275: weight grid, nested by person), new people. Per order and base the V6Fr outputs of the control and of each arm
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
ARMS = [f"nm{a}_w{w}" for a in ("A", "B") for w in os.environ.get("WS", "0.1,0.2,0.3,0.5").split(",")]; SCHEMES = ("within", "full", "runs")


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

WS = [w for w in os.environ.get("WS", "0.1,0.2,0.3,0.5").split(",")]; rng = np.random.default_rng(0)
by = {s: [o for o in ORD if o.rstrip("0123456789") == s] for s in SCHEMES}; by = {s: v for s, v in by.items() if v}
OUT = []
def say(x=""):
    print(x); OUT.append(x)
def G(a, w, suf):  # per scheme: orders x 10 people, arm minus control (raw: vs ctl raw; filtered: vs ctl raw minus ctl filter effect, i.e. filtered arm - filtered ctl)
    k = f"nm{a}_w{w}" + suf
    return {s: np.stack([R[o, k] - (R[o, "ctl+filter"] if suf else 0) for o in v]) for s, v in by.items()}
def pooled(D, pp=None):  # schemes equal weight, orders averaged, per person (10,)
    return np.mean([d.mean(0) for d in D.values()], 0)
def interval(fn):  # people-resampled 80 % interval of the statistic fn(person index array)
    bt = [fn(rng.integers(0, 10, 10)) for _ in range(4000)]; return np.percentile(bt, [10, 90])
for suf, lab in (("", "raw"), ("+filter", "count-filtered")):
    say(f"=== {lab}: arm minus control (new people; schemes equal weight; people-resampled 80 % interval; chain at weight w x base) ===")
    for a in ("A", "B"):
        Ds = {w: G(a, w, suf) for w in WS}; PM = np.stack([pooled(Ds[w]) for w in WS])  # (W, 10)
        say(f"-- {a}: w curve (mean over people, [80 %], people up)")
        for j, w in enumerate(WS):
            lo, hi = interval(lambda pp, j=j: PM[j][pp].mean())
            say(f"   w={w:4s} {PM[j].mean():+.2f} [{lo:+.2f}, {hi:+.2f}] up {int((PM[j] > 0).sum())}/10 | " + " ".join(f"{s} {d.mean():+.2f}" for s, d in Ds[w].items()))
        # nested by person: w chosen on the other nine people (people x orders mean; schemes equal weight), held-out person scored at that w
        def pick(idx):  # idx: person indices available for selection
            return int(np.argmax(PM[:, idx].mean(1)))
        sel = [pick([q for q in range(10) if q != p]) for p in range(10)]; nested = np.array([PM[sel[p], p] for p in range(10)])
        sch = {s: np.mean([np.stack([Ds[WS[sel[p]]][s].mean(0)[p] for p in range(10)])]) for s in by}
        def nested_fn(pp):  # resample people, redo the selection inside the resample (people-nested)
            vals = []
            for i, p in enumerate(pp):
                rest = [r for r in range(len(pp)) if r != i]; j = int(np.argmax(PM[:, pp[rest]].mean(1))); vals.append(PM[j, p])
            return np.mean(vals)
        lo, hi = interval(nested_fn); bf = int(np.argmax(PM.mean(1)))
        say(f"   NESTED {nested.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] people up {int((nested > 0).sum())}/10 | selected w per held-out person: {[WS[x] for x in sel]} | modal {max(set(sel), key=sel.count) and WS[max(set(sel), key=sel.count)]}")
        say("   nested per scheme: " + " ".join(f"{s} {sch[s]:+.2f}" for s in by) + f" | best fixed w={WS[bf]} {PM[bf].mean():+.2f} (optimistic)")
        say("   nested people: " + " ".join(f"{x:+.2f}" for x in nested))
open(f"{REPO}/evidence/newmember_15nov/s275.txt", "w").write("\n".join(OUT) + "\n")
np.savez(f"{REPO}/runs/newmember/score_s275.npz", **{f"{o}|{k}": v for (o, k), v in R.items()})
