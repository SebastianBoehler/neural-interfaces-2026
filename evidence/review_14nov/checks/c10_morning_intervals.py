"""morning_table.py prints an 80 % bootstrap interval over ORDERS of the ten-person mean. The same ten people sit in every
order, so that interval leaves out person spread. Recomputes, per scheme: interval over orders (as printed), over people
(orders averaged first), and over both; and the new-people mean restricted to the orders the returning rows have."""
import glob, os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); NPZ = REPO + "/runs/leo_npz"; RET = REPO + "/runs/retv6"; rng = np.random.default_rng(0)


def mb(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); assert len(b) == 6; return np.mean([z[f"{bn}|{arm}"] for bn in b], 0)


PAIRS = {}
for d, tag in ((REPO + "/runs", "QF"), (NPZ, "RF"), (NPZ, "KF")):
    for f in sorted(glob.glob(f"{d}/shufcombo_*_ocu_gC5_all_ge_hb_ts_gp0_ttt18*L.npz")):
        o = os.path.basename(f).split("_")[1]; g = glob.glob(f"{d}/shufcombo_{o}_ocu_s15_all_ge_hb_ts_gp0_ttt18{o}{tag}.npz")
        if g and f.endswith(f"ttt18{o}L.npz"):
            PAIRS[o] = (f, g[0])
new = {o: {"v6f": mb(f, "V6Fr") - mb(f, "P0r"), "v6.1": mb(g, "V6Fr") - mb(g, "P0r")} for o, (f, g) in PAIRS.items()}
ret = {os.path.basename(f)[:-4].split("_")[-1] for f in glob.glob(f"{RET}/g1_s15_QF_seen_session_*.npz")}
pc = lambda x: np.percentile(x, [10, 90])
for sch in ("within", "full", "runs"):
    os_ = sorted(o for o in new if o.startswith(sch)); k = len(os_)
    for v in ("v6f", "v6.1"):
        M = np.array([new[o][v] for o in os_])  # orders x people
        a = pc([M[rng.integers(0, k, k)].mean() for _ in range(4000)]); b = pc([M[:, rng.integers(0, 10, 10)].mean() for _ in range(4000)])
        c = pc([M[rng.integers(0, k, k)][:, rng.integers(0, 10, 10)].mean() for _ in range(4000)])
        sub = [o for o in os_ if o in ret]
        print(f"{sch:6s} {v:5s} {k:2d} orders: mean {M.mean():+.2f} | over orders (as printed) [{a[0]:+.2f}, {a[1]:+.2f}] | over people [{b[0]:+.2f}, {b[1]:+.2f}] | both [{c[0]:+.2f}, {c[1]:+.2f}] | "
              f"per person {' '.join(f'{x:+.1f}' for x in M.mean(0))} | on the {len(sub)} orders the returning rows have: {np.mean([new[o][v].mean() for o in sub]):+.2f}")
    D = np.array([new[o]["v6.1"] - new[o]["v6f"] for o in os_]); b = pc([D[:, rng.integers(0, 10, 10)].mean() for _ in range(4000)]); a = pc([D[rng.integers(0, k, k)].mean() for _ in range(4000)])
    print(f"{sch:6s} v6.1 - v6f: {D.mean():+.2f} | over orders [{a[0]:+.2f}, {a[1]:+.2f}] | over people [{b[0]:+.2f}, {b[1]:+.2f}] | without P10 {np.delete(D, 9, 1).mean():+.2f}")
