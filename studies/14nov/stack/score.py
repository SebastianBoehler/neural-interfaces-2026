"""STACK scoring (ledger s242), new people. Per order and base, V6Fr of each arm file runs/chainout/s242<arm>_<order>.npz
(SAVE_OUT, recording layout), raw and through the detector-gated count filter (rundetect/package.py: latch + J-wprior urn,
emitted output only). Every cell is the per-person change against ctl raw (s15 on the same LP labels, no view, no filter),
mean over bases. Intervals resample people and orders (two-way bootstrap; orders within scheme, schemes pooled with equal
weight); 'people up' counts person means over orders. Optional reference REF=runs/chainout/v62hL_{order}.npz (LF s15).
Usage: [ARMS=stk1] [SFX=d1] python studies/14nov/stack/score.py order [order ...]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); sys.argv, ORD = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, REPO + "/studies/14nov/rundetect")
import package as K  # noqa: E402  (latch, gate, J, C)
ARMS = os.environ.get("ARMS", "stk1").split(","); REF = os.environ.get("REF"); SFX = os.environ.get("SFX", "")  # s243: d1, d2 = draws
SCHEMES = ("within", "full", "runs")


def outs(f, order, hc, lat):
    z = np.load(f); bases = sorted({k.split("|")[0] for k in z.files if k.endswith("|V6Fr")}); raw, fil = [], []
    for bn in bases:
        P = np.full((K.C.N, 3), 1 / 3); P[K.ROWS] = z[f"{bn}|V6Fr"].astype(np.float64)
        raw.append(K.C.A.SC.person(P[K.ROWS], K.PROTO)); fil.append(K.C.A.SC.person(K.gate(P, K.J.filt_joint(P, hc, "urn"), order, lat)[K.ROWS], K.PROTO))
    return bases, np.mean(raw, 0), np.mean(fil, 0)


R = {}
for order in ORD:
    lat = K.latch(order); hc = K.J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
    bc, c0, c1 = outs(f"{REPO}/runs/chainout/s242ctl{SFX}_{order}.npz", order, hc, lat); R[order, "ctl+filter"] = c1 - c0
    for arm in ARMS:
        ba, a0, a1 = outs(f"{REPO}/runs/chainout/s242{arm}{SFX}_{order}.npz", order, hc, lat); assert ba == bc, (arm, order, ba, bc)
        R[order, arm] = a0 - c0; R[order, arm + "+filter"] = a1 - c0
    if REF:
        _, r0, r1 = outs(REPO + "/" + REF.format(order=order), order, hc, lat); R[order, "ref(LF s15)"] = r0 - c0; R[order, "ref+filter"] = r1 - c0
    print(f"  {order:8s} latched {sum(w is not None for w in lat.values())}/{len(lat)} | " + " ".join(f"{k} {v.mean():+.2f}" for (o, k), v in R.items() if o == order), flush=True)

rng = np.random.default_rng(0); keys = list(dict.fromkeys(k for _, k in R))
print("\nchange against ctl raw, new people: mean | per scheme | [80 % interval over people and orders] | people up")
for k in keys:
    by = {s: [o for o in ORD if o.rstrip("0123456789") == s] for s in SCHEMES}; by = {s: v for s, v in by.items() if v}
    D = {s: np.stack([R[o, k] for o in v]) for s, v in by.items()}  # (orders, people)
    pt = np.mean([d.mean() for d in D.values()]); bt = []
    for _ in range(4000):
        pp = rng.integers(0, 10, 10); bt.append(np.mean([d[rng.integers(0, len(d), len(d))][:, pp].mean() for d in D.values()]))
    pm = np.mean([d.mean(0) for d in D.values()], 0); lo, hi = np.percentile(bt, [10, 90])
    print(f"  {k:14s} {pt:+.2f} | " + " ".join(f"{s} {d.mean():+.2f} ({len(d)})" for s, d in D.items()) + f" | [{lo:+.2f}, {hi:+.2f}] | {int((pm > 0).sum())}/10")
np.savez(f"{REPO}/runs/s242{SFX}_{'_'.join(ARMS)}.npz", **{f"{o}|{k}": v for (o, k), v in R.items()})
