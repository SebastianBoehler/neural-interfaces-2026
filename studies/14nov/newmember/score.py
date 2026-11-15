"""NEWMEMBER scoring (ledger s272), new people. Per order and base the V6Fr outputs of the control and of each arm
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
ARMS = os.environ["ARMS"].split(","); SCHEMES = ("within", "full", "runs")


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
        if arm.startswith("nmH"):  # s273 seeds halved: only the bases whose pool is not inside seeds 0-2 (pool, s345) may change; the others must be bit-equal
            ch = {k.split("|")[0]: not np.array_equal(za[k], zc[k]) for k in za.files if k.endswith("|V6Fr")}
            assert all(ch[b] == (b in ("B2_pool", "B2_s345")) for b in ch), f"{arm} {order}: changed bases {ch}"
        elif arm.startswith("nmDnone"):
            assert all(bits), f"{arm} {order}: DROP=none is not bit-identical to the control"
        elif arm.startswith("nmZ"):
            assert all(bits), f"{arm} {order}: zero-weight slot is not bit-identical to the control"
        else:
            assert not any(bits), f"{arm} {order}: arm output equals the control for some base (the new slot has no effect)"
        R[order, arm] = a0 - c0; R[order, arm + "+filter"] = a1 - c0
    print(f"  {order:8s} latched {sum(w is not None for w in lat.values())}/{len(lat)} ctl raw {c0.mean():.2f} | " + " ".join(f"{k} {v.mean():+.2f}" for (o, k), v in R.items() if o == order), flush=True)
rng = np.random.default_rng(0); keys = list(dict.fromkeys(k for _, k in R))
print("\nchange against ctl raw (arm+filter rows: arm filtered - ctl raw; compare with ctl+filter), new people: mean | per scheme | [80 % interval] | people up")
for k in keys:
    by = {s: [o for o in ORD if o.rstrip("0123456789") == s] for s in SCHEMES}; by = {s: v for s, v in by.items() if v}
    D = {s: np.stack([R[o, k] for o in v]) for s, v in by.items()}
    pt = np.mean([d.mean() for d in D.values()]); bt = []
    for _ in range(4000):
        pp = rng.integers(0, 10, 10); bt.append(np.mean([d[rng.integers(0, len(d), len(d))][:, pp].mean() for d in D.values()]))
    pm = np.mean([d.mean(0) for d in D.values()], 0); lo, hi = np.percentile(bt, [10, 90])
    ups = " ".join(f"{s} {int((d.mean(0) > 0).sum())}/10" for s, d in D.items())
    print(f"  {k:22s} {pt:+.2f} | " + " ".join(f"{s} {d.mean():+.2f} ({len(d)})" for s, d in D.items()) + f" | [{lo:+.2f}, {hi:+.2f}] | {int((pm > 0).sum())}/10 | people up per scheme: {ups}")
# contrast of each arm to the control, same machinery (arm - ctl, raw and filtered)
print("\narm minus ctl (the s272 reading): raw and with the count filter, both against ctl's own raw/filtered output")
for arm in ARMS:
    for suf in ("", "+filter"):
        c = "ctl+filter" if suf else None
        D = {s: np.stack([R[o, arm + suf] - (R[o, c] if c else 0) for o in v]) for s in SCHEMES if (v := [o for o in ORD if o.rstrip("0123456789") == s])}
        pt = np.mean([d.mean() for d in D.values()]); bt = []
        for _ in range(4000):
            pp = rng.integers(0, 10, 10); bt.append(np.mean([d[rng.integers(0, len(d), len(d))][:, pp].mean() for d in D.values()]))
        pm = np.mean([d.mean(0) for d in D.values()], 0); lo, hi = np.percentile(bt, [10, 90])
        print(f"  {arm}{suf or ' raw':10s} {pt:+.2f} [{lo:+.2f}, {hi:+.2f}] people up {int((pm > 0).sum())}/10 | " + " ".join(f"{s} {d.mean():+.2f} up {int((d.mean(0) > 0).sum())}/10" for s, d in D.items()))
np.savez(f"{REPO}/runs/newmember/score_{'_'.join(ARMS)}.npz", **{f"{o}|{k}": v for (o, k), v in R.items()})
