"""SOFT OFFSETS scoring (ledger s251 (2)), new people, V6Fr, six bases. Per order: runs/softfp/so_<order>_zero.npz must equal
runs/chainout/s242r15_<order>.npz exactly (the hook's plumbing); then kern / top10 against s242r15, raw and with both
through the s228c detector-gated count filter (latchJ, the current emission; primary). Intervals resample people and
orders. Usage: [FORMS=kern,top10] [ZF=zero] python score_off.py [orders]   (TSEM local templates: FORMS=tloc ZF=tzero)"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/../rundetect")
from softcount import boot  # noqa: E402
import package as K  # noqa: E402
C, J = K.C, K.J; SC = C.A.SC; FORMS = os.environ.get("FORMS", "kern,top10").split(","); ZF = os.environ.get("ZF", "zero")  # TSEM (3): FORMS=tloc ZF=tzero
ORDERS = sys.argv[1:] or [f"{s}{i}" for s in ("full", "within", "runs") for i in range(4)]


def main():
    rng = np.random.default_rng(0); R = {}
    for order in ORDERS:
        ref = np.load(f"{C.REPO}/runs/chainout/s242r15_{order}.npz"); bases = sorted({k.split("|")[0] for k in ref.files if k.endswith("|V6Fr")})
        zz = np.load(f"{C.REPO}/runs/softfp/so_{order}_{ZF}.npz")
        for bn in bases:
            assert np.array_equal(zz[f"{bn}|V6Fr"], ref[f"{bn}|V6Fr"]), f"{ZF} form differs from s242r15 ({order} {bn})"
        lat = K.latch(order); hc = J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
        sc = lambda z: np.mean([[SC.person(P, C.p0), SC.person(K.gate(P, J.filt_joint(P, hc, "urn"), order, lat), C.p0)]  # noqa: E731
                                for P in (z[f"{bn}|V6Fr"].astype(np.float64) for bn in bases)], 0)
        r0 = sc(ref)
        for fm in FORMS:
            r1 = sc(np.load(f"{C.REPO}/runs/softfp/so_{order}_{fm}.npz")); R[order, fm, "raw"] = r1[0] - r0[0]; R[order, fm, "latchJ"] = r1[1] - r0[1]
        print(f"  {order:8s} {ZF} form == s242r15 | " + " | ".join(f"{fm}: raw {R[order, fm, 'raw'].mean():+.2f} latchJ {R[order, fm, 'latchJ'].mean():+.2f}" for fm in FORMS), flush=True)
    print("SUMMARY (change against s242r15, mean over orders [80 % interval, people and orders resampled] people up)")
    for fm in FORMS:
        for arm in ("latchJ", "raw"):
            cells = []
            for s in ("full", "within", "runs"):
                d = np.array([R[o, fm, arm] for o in ORDERS if o.startswith(s)])
                if len(d):
                    b = boot(d, rng); cells.append(f"{s} {d.mean():+.2f} [{b[0]:+.2f}, {b[1]:+.2f}] {int((d.mean(0) > 0).sum())}/10")
            print(f"  {fm:6s} {arm:6s} " + " | ".join(cells), flush=True)
    np.savez(f"{C.REPO}/runs/softfp/score_off_{'_'.join(FORMS)}.npz", **{"|".join(k): v for k, v in R.items()})


if __name__ == "__main__":
    main()
