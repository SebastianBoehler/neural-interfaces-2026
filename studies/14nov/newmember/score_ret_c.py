"""s276 scoring, returning people (P04, P10; seen_session). Control = runs/chainout/ret_<order>.npz (ER, the s214/s237 outputs); arms = runs/chainout/ret_<ARM>_<order>.npz
from ret_c.py (key <order>|<base>|ER). Per order and base: person scores raw and through the detector-gated count filter (studies/14nov/rundetect/package.py gate, latchJ);
change arm - control with the same filter, mean over the base set (six, B2_pool, B2_s345), then mean over orders within scheme, schemes equal weight.
P04 and P10 are printed separately, with their mean and the returning weighting 0.2 P04 + 0.8 P10; no bootstrap (two people): the spread is over the order draws.
C0 (DROP=none HALF=none) must equal the control to 1e-5. Usage: ARMS=C1,C2 python studies/14nov/newmember/score_ret_c.py ORDER [ORDER ...]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); sys.argv, ORD = sys.argv[:1], sys.argv[1:]
os.environ["PROTO"] = "seen_session"; sys.path.insert(0, REPO + "/studies/14nov/rundetect")
import package as K  # noqa: E402
ARMS = os.environ["ARMS"].split(","); SETS = {"six": None, "pool": ("B2_pool",), "s345": ("B2_s345",)}; SCH = ("within", "full", "runs")


def outs(z, order, hc, lat):
    raw, fil = {}, {}
    for bn in sorted({k.split("|")[1] for k in z.files}):
        P = np.full((K.C.N, 3), 1 / 3); P[K.ROWS] = z[f"{order}|{bn}|ER"].astype(np.float64)
        raw[bn] = K.C.A.SC.person(P[K.ROWS], K.PROTO); fil[bn] = K.C.A.SC.person(K.gate(P, K.J.filt_joint(P, hc, "urn"), order, lat)[K.ROWS], K.PROTO)
    return raw, fil


def avg(d, sel):
    return np.mean([v for b, v in d.items() if sel is None or b in sel], 0)


R, CT = {}, {}
who = sorted(dict.fromkeys(K.C.A.L.Z[f"person|{K.PROTO}"])); assert who == ["P04", "P10"], who
for order in ORD:
    lat = K.latch(order); hc = K.J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
    zc = np.load(f"{REPO}/runs/chainout/ret_{order}.npz"); c0, c1 = outs(zc, order, hc, lat)
    for arm in ARMS:
        za = np.load(f"{REPO}/runs/chainout/ret_{arm}_{order}.npz"); a0, a1 = outs(za, order, hc, lat); assert set(a0) == set(c0), (arm, order)
        if arm == "C0":
            assert all(np.abs(za[f"{order}|{b}|ER"] - zc[f"{order}|{b}|ER"]).max() < 1e-5 for b in a0), "C0 differs from the control"
        for sn, sel in SETS.items():
            R[order, arm, "raw", sn] = avg(a0, sel) - avg(c0, sel); R[order, arm, "filtered", sn] = avg(a1, sel) - avg(c1, sel)
    print(f"  {order:8s} ctl raw P04 {avg(c0, None)[0]:.2f} P10 {avg(c0, None)[1]:.2f} | " + " ".join(f"{a} {R[order, a, 'raw', 'six'].round(2)}" for a in ARMS), flush=True)
print("\narm minus control, returning people, per-person change (P04, P10), mean over orders (schemes equal weight); [order spread = sd of the order draws / sqrt(n)]")
for arm in ARMS:
    for kind in ("raw", "filtered"):
        for sn in SETS:
            if sn != "six" and arm in ("C1", "C0"):
                continue
            by = {s: np.mean([R[o, arm, kind, sn] for o in ORD if o.rstrip("0123456789") == s], 0) for s in SCH if any(o.rstrip("0123456789") == s for o in ORD)}
            m = np.mean(list(by.values()), 0); allr = np.stack([R[o, arm, kind, sn] for o in ORD]); se = allr.std(0, ddof=1) / np.sqrt(len(ORD)) if len(ORD) > 1 else np.zeros(2)
            print(f"  {arm:3s} {kind:8s} @{sn:5s} P04 {m[0]:+.2f} (+-{se[0]:.2f}) P10 {m[1]:+.2f} (+-{se[1]:.2f}) | mean {m.mean():+.2f} | 0.2/0.8 {0.2 * m[0] + 0.8 * m[1]:+.2f} | " + " ".join(f"{s} {v.mean():+.2f}" for s, v in by.items()))
np.savez(f"{REPO}/runs/newmember/score_ret_c_{'_'.join(ARMS)}.npz", **{"|".join(k): v for k, v in R.items()})
