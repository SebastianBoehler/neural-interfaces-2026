"""SOFT STACK (ledger s251 amendment 2, item 4), new people, V6Fr, six bases: the soft-offset chain (runs/softfp/so_<order>_kern.npz)
with the normalised soft count (m15; m20 reported) on its output before the s228c latch and latchJ after it, against the
current emission latchJ(s242r15). Also the parts: offsets alone through latchJ, and the soft count alone on s242r15.
Intervals resample people and orders. Usage: [OFF=kern] python stack.py [orders]"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import softcount as S  # noqa: E402
K, C, J, fpw = S.K, S.C, S.J, S.fpw; SC = C.A.SC; OFF = os.environ.get("OFF", "kern"); MS = (15, 20)
ORDERS = sys.argv[1:] or [f"{s}{i}" for s in ("full", "within", "runs") for i in range(4)]


def main():
    rng = np.random.default_rng(0); R = {}
    for order in ORDERS:
        lat = K.latch(order); hc = J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO); W = fpw.all_weights(order)
        ld = lambda f: np.stack([np.load(f)[f"{bn}|V6Fr"].astype(np.float64) for bn in BASES])  # noqa: E731
        ref = np.load(f"{C.REPO}/runs/chainout/s242r15_{order}.npz"); BASES = sorted({k.split("|")[0] for k in ref.files if k.endswith("|V6Fr")})
        P0, P1 = ld(f"{C.REPO}/runs/chainout/s242r15_{order}.npz"), ld(f"{C.REPO}/runs/softfp/so_{order}_{OFF}.npz")
        lj = lambda Pb: np.stack([K.gate(P, J.filt_joint(P, hc, "urn"), order, lat) for P in Pb])  # noqa: E731
        sc = lambda X: np.mean([SC.person(x, C.p0) for x in X], 0)  # noqa: E731
        L0, L1 = lj(P0), lj(P1); base = sc(L0); R[order, "offsets"] = sc(L1) - base
        for M in MS:
            for nm, Pb, Lb in (("count", P0, L0), ("stack", P1, L1)):
                Sf = S.scf(Pb, W, M); R[order, f"{nm} m{M}"] = sc(np.stack([K.gate(a, b, order, lat) for a, b in zip(Sf, Lb)])) - base
        print(f"  {order:8s} " + " | ".join(f"{k} {R[order, k].mean():+.2f}" for k in ["offsets"] + [f"{n} m{M}" for M in MS for n in ("count", "stack")]), flush=True)
    print(f"SUMMARY (against latchJ(s242r15); offsets form {OFF}; mean over orders [80 % interval, people and orders resampled] people up)")
    for k in ["offsets"] + [f"{n} m{M}" for M in MS for n in ("count", "stack")]:
        cells = []
        for s in ("full", "within", "runs"):
            d = np.array([R[o, k] for o in ORDERS if o.startswith(s)])
            if len(d):
                b = S.boot(d, rng); cells.append(f"{s} {d.mean():+.2f} [{b[0]:+.2f}, {b[1]:+.2f}] {int((d.mean(0) > 0).sum())}/10")
        print(f"  {k:12s} " + " | ".join(cells), flush=True)
    np.savez(f"{C.REPO}/runs/softfp/stack_{OFF}.npz", **{f"{o}|{k}": v for (o, k), v in R.items()})


if __name__ == "__main__":
    main()
