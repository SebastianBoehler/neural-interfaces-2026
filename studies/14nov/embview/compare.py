"""EMBVIEW scoring (ledger s234): V6Fr with EXTRA_VIEW=runs/views_emb/<name>.npz against the same chain without it
(runs/shufcombo_<order>_ocu_gC5_all_ge_hb_ts_gp0_ttt18<order>L.npz), new people, mean over the six bases.
Per order: mean [80 % person interval]; then the mean over orders, orders up, and people up (person means over orders).
Usage: python studies/14nov/embview/compare.py NAME [NAME ...]"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../..")
ORDERS = ("within0", "within1", "full0", "full1", "runs0", "runs1")
BASES = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")


def delta(name, o, arm="V6Fr"):
    f = f"{REPO}/runs/shufcombo_{o}_ocu_gC5_all_ge_hb_ts_gp0_ttt18{o}L"; a, b = np.load(f"{f}_xv{name}.npz"), np.load(f"{f}.npz")
    for bn in BASES:  # the reference arm must be identical: the view enters only the V6* arms
        assert np.allclose(a[f"{bn}|P0r"], b[f"{bn}|P0r"]), (name, o, bn)
    return np.mean([a[f"{bn}|{arm}"] - b[f"{bn}|{arm}"] for bn in BASES], 0)


def band(d, rng):
    bt = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]; return np.percentile(bt, [10, 90])


for name in sys.argv[1:]:
    rng = np.random.default_rng(0); D = {}
    for o in ORDERS:
        if os.path.exists(f"{REPO}/runs/shufcombo_{o}_ocu_gC5_all_ge_hb_ts_gp0_ttt18{o}L_xv{name}.npz"):
            D[o] = delta(name, o)
    if not D:
        print(f"{name}: no files yet"); continue
    cells = []
    for o, d in D.items():
        lo, hi = band(d, rng); v6r = delta(name, o, "V6r").mean(); cells.append(f"{o} {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] (fallback {v6r:+.2f})")
    pm = np.mean(list(D.values()), 0); lo, hi = band(pm, rng)
    print(f"{name}: " + "; ".join(cells))
    print(f"{name}: mean over {len(D)} orders {pm.mean():+.2f} [{lo:+.2f}, {hi:+.2f}], orders up {sum(d.mean() > 0 for d in D.values())}/{len(D)}, "
          f"people up {(pm > 0).sum()}/10 | per person " + " ".join(f"{v:+.2f}" for v in pm), flush=True)
