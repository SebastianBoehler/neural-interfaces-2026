"""PV2/PV3 summary: change against ER per arm, six-base mean per order, then per scheme and over all orders; P04 and P10
separately; 80 % bootstrap interval over orders (and over orders x people for the mean); orders > 0 for P10 and the mean.
Usage: python studies/14nov/pview/summary.py runs/pview/chain_<orders>.npz"""
import sys
import numpy as np

z = np.load(sys.argv[1]); keys = [k.split("|") for k in z.files]
orders = list(dict.fromkeys(k[0] for k in keys)); bases = list(dict.fromkeys(k[1] for k in keys)); arms = list(dict.fromkeys(k[2] for k in keys))
assert len(bases) == 6, f"{len(bases)} bases (smoke file?)"
rng = np.random.default_rng(0)


def d(o, a):  # (2,) six-base mean change against ER, (P04, P10)
    return np.mean([z[f"{o}|{b}|{a}"] - z[f"{o}|{b}|ER"] for b in bases], 0)


def band(x):
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]; return np.percentile(bs, [10, 90])


print(f"change against ER (eye view + rest member, s214), six-base mean; {len(orders)} orders: {','.join(orders)}")
print("arm   scheme  mean [80% over orders]   P04 [80%]            P10 [80%]            P10>0  mean>0")
for a in [a for a in arms if a != "ER"]:
    for sc in ("within", "full", "runs", "all"):
        oo = [o for o in orders if sc == "all" or o.rstrip("0123456789") == sc]
        if not oo:
            continue
        D = np.array([d(o, a) for o in oo]); m = D.mean(1)
        b0, b4, b10 = band(m), band(D[:, 0]), band(D[:, 1])
        print(f"{a:5s} {sc:7s} {m.mean():+.2f} [{b0[0]:+.2f},{b0[1]:+.2f}]   {D[:, 0].mean():+.2f} [{b4[0]:+.2f},{b4[1]:+.2f}]   "
              f"{D[:, 1].mean():+.2f} [{b10[0]:+.2f},{b10[1]:+.2f}]   {(D[:, 1] > 0).sum():2d}/{len(oo)}  {(m > 0).sum():2d}/{len(oo)}")
er = np.array([np.mean([z[f"{o}|{b}|ER"] for b in bases], 0) for o in orders])
print(f"ER absolute, mean over orders: P04 {er[:, 0].mean():.2f} P10 {er[:, 1].mean():.2f}")
