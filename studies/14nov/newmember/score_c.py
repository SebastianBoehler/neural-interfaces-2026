"""s276 scoring, new people (own copy of score.py, which is not edited). Per order and base the V6Fr outputs of the control (runs/chainout/nm_ctl_<order>.npz)
and of each arm (nm_<ARM>_<order>.npz from run_c.py), raw and through the detector-gated count filter (as score.py). Cell = per-person change arm - ctl, averaged
over a base set: the six bases, and for the halving arms B2_pool and B2_s345 separately (the bases where seeds 3-5 exist; the others are bit-equal for C4).
80 % intervals resample people and orders (orders within scheme, schemes equal weight); 'people up' = person means over orders.
Arm checks: C0 (DROP=none HALF=none) bit-equal to ctl in every base; C4 (halve all, no drop) changes exactly B2_pool and B2_s345; other arms change every base.
Usage: ARMS=C1,C2,C3,C4 python studies/14nov/newmember/score_c.py ORDER [ORDER ...]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); sys.argv, ORD = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, REPO + "/studies/14nov/rundetect")
import package as K  # noqa: E402
ARMS = os.environ["ARMS"].split(","); SCHEMES = ("within", "full", "runs"); SETS = {"six": None, "pool": ("B2_pool",), "s345": ("B2_s345",)}


def outs(f, order, hc, lat):
    z = np.load(f); bases = sorted({k.split("|")[0] for k in z.files if k.endswith("|V6Fr")}); raw, fil = {}, {}
    for bn in bases:
        P = np.full((K.C.N, 3), 1 / 3); P[K.ROWS] = z[f"{bn}|V6Fr"].astype(np.float64)
        raw[bn] = K.C.A.SC.person(P[K.ROWS], K.PROTO); fil[bn] = K.C.A.SC.person(K.gate(P, K.J.filt_joint(P, hc, "urn"), order, lat)[K.ROWS], K.PROTO)
    return bases, raw, fil


def avg(d, sel):
    return np.mean([v for b, v in d.items() if sel is None or b in sel], 0)


def path(name, order):
    return f"{REPO}/runs/chainout/nm_{name}_{order}.npz"


R = {}
for order in ORD:
    lat = K.latch(order); hc = K.J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
    bc, c0, c1 = outs(path("ctl", order), order, hc, lat); zc = np.load(path("ctl", order))
    for sn, sel in SETS.items():
        R[order, f"ctl+filter@{sn}"] = avg(c1, sel) - avg(c0, sel)
    for arm in ARMS:
        ba, a0, a1 = outs(path(arm, order), order, hc, lat); assert ba == bc, (arm, order, ba, bc)
        za = np.load(path(arm, order)); ch = {k.split("|")[0]: not np.array_equal(za[k], zc[k]) for k in za.files if k.endswith("|V6Fr")}
        if arm == "C0":
            assert not any(ch.values()), f"C0 {order}: DROP=none HALF=none is not bit-identical to the control {ch}"
        elif arm == "C4":
            assert all(ch[b] == (b in ("B2_pool", "B2_s345")) for b in ch), f"C4 {order}: changed bases {ch}"
        else:
            assert all(ch.values()), f"{arm} {order}: an arm base equals the control {ch}"
        for sn, sel in SETS.items():
            R[order, f"{arm}@{sn}"] = avg(a0, sel) - avg(c0, sel); R[order, f"{arm}+filter@{sn}"] = avg(a1, sel) - avg(c0, sel)
    print(f"  {order:8s} ctl raw {avg(c0, None).mean():.2f} | " + " ".join(f"{k} {v.mean():+.2f}" for (o, k), v in R.items() if o == order and k.endswith("@six") and "ctl" not in k), flush=True)
rng = np.random.default_rng(0)


def stat(key, sub=None):
    D = {s: np.stack([R[o, key] - (R[o, sub] if sub else 0) for o in v]) for s in SCHEMES if (v := [o for o in ORD if o.rstrip("0123456789") == s])}
    pt = np.mean([d.mean() for d in D.values()]); bt = []
    for _ in range(4000):
        pp = rng.integers(0, 10, 10); bt.append(np.mean([d[rng.integers(0, len(d), len(d))][:, pp].mean() for d in D.values()]))
    pm = np.mean([d.mean(0) for d in D.values()], 0); lo, hi = np.percentile(bt, [10, 90])
    return f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}] people up {int((pm > 0).sum())}/10 | " + " ".join(f"{s} {d.mean():+.2f} up {int((d.mean(0) > 0).sum())}/10" for s, d in D.items())


print("\narm minus ctl, new people (raw: arm raw - ctl raw; filtered: arm filtered - ctl filtered), base set after @")
for arm in ARMS:
    for sn in SETS:
        if sn != "six" and arm in ("C1",):
            continue
        print(f"  {arm:3s} raw      @{sn:5s} {stat(f'{arm}@{sn}')}")
        print(f"  {arm:3s} filtered @{sn:5s} {stat(f'{arm}+filter@{sn}', f'ctl+filter@{sn}')}")
np.savez(f"{REPO}/runs/newmember/score_c_{'_'.join(ARMS)}.npz", **{f"{o}|{k}": v for (o, k), v in R.items()})
