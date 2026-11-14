"""s233 step 3 table: V6Fr (v6f's combination, new people, mean over the six bases) with EXTRA_VIEW=<name> minus the
regular file of the same order; per order mean, 80 % person interval (4000 resamples of the 10 people), people up;
then the mean over orders and people up on the person means over orders (the kill rule's numbers).
Usage: python studies/14nov/extview/compare.py mivma mavr shin2"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
ORDERS = [f"{s}{k}" for s in ("within", "full", "runs") for k in range(2)]
BASES = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")


def v6(path):
    z = np.load(path); return np.mean([z[f"{b}|V6Fr"] for b in BASES], 0), np.mean([z[f"{b}|V6r"] for b in BASES], 0)


def band(d, rng):
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)], [10, 90])


for nm in sys.argv[1:]:
    rng = np.random.default_rng(0); D, Dr = [], []
    for o in ORDERS:
        f = f"{REPO}/runs/shufcombo_{o}_ocu_gC5_all_ge_hb_ts_gp0_ttt18{o}L"
        if not os.path.exists(f + f"_xv{nm}.npz"):
            print(f"{nm} {o}: missing"); continue
        (a, ar), (b, br) = v6(f + f"_xv{nm}.npz"), v6(f + ".npz"); d = a - b; D.append(d); Dr.append(ar - br); lo, hi = band(d, rng)
        print(f"{nm:6s} {o:8s} V6Fr {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] up {(d > 0).sum()}/10 | without P10 {d[:9].mean():+.2f} | "
              f"fallback V6r {(ar - br).mean():+.2f} | people " + " ".join(f"{v:+.1f}" for v in d), flush=True)
    if len(D) == len(ORDERS):
        m = np.mean(D, 0); lo, hi = band(m, rng)
        print(f"{nm:6s} MEAN6    V6Fr {m.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] people up {(m > 0).sum()}/10, orders up {sum(d.mean() > 0 for d in D)}/6 | "
              f"V6r {np.mean(Dr):+.2f} | kill rule (>= +0.3 and >= 7/10): {'KEPT' if m.mean() >= 0.3 and (m > 0).sum() >= 7 else 'not kept'}", flush=True)
