"""Read-only re-reading of the morning-table inputs (runs/leo_npz, runs/, runs/retv6): per-person gains, a person
bootstrap (the table's own interval resamples orders only), and sealed-population weightings. Writes nothing."""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); NPZ = REPO + "/runs/leo_npz"; RET = REPO + "/runs/retv6"
rng = np.random.default_rng(0)


def mb(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); assert len(b) == 6
    return np.mean([z[f"{bn}|{arm}"] for bn in b], 0)


PAIRS = {}
for d, tag in ((REPO + "/runs", "QF"), (NPZ, "RF"), (NPZ, "KF")):
    for f in sorted(glob.glob(f"{d}/shufcombo_*_ocu_gC5_all_ge_hb_ts_gp0_ttt18*L.npz")):
        o = os.path.basename(f).split("_")[1]; g = glob.glob(f"{d}/shufcombo_{o}_ocu_s15_all_ge_hb_ts_gp0_ttt18{o}{tag}.npz")
        if g:
            PAIRS[o] = (f, g[0])
new = {o: {"v6f": mb(f, "V6Fr") - mb(f, "P0r"), "v6.1": mb(g, "V6Fr") - mb(g, "P0r"), "fallback": mb(f, "V6r") - mb(f, "P0r"),
           "loss": mb(f, "P0r") - mb(f, "P0rec"), "abs": mb(f, "P0rec")} for o, (f, g) in PAIRS.items()}
ret = {}
for f in sorted(glob.glob(f"{RET}/g1_s15_QF_seen_session_*.npz")):
    o = os.path.basename(f)[:-4].split("_")[-1]; z = np.load(f); b = sorted({k.split("|")[1] for k in z.files})
    ret[o] = {k: np.mean([z[f"{o}|{bn}|{k}"] - z[f"{o}|{bn}|base"] for bn in b], 0) for k in ("E", "R", "ER", "F", "ERF")}
    ret[o]["abs"] = np.mean([z[f"{o}|{bn}|base"] for bn in b], 0)
print("orders new:", sorted(new), "\norders returning:", sorted(ret))
PPL = [f"P{i:02d}" for i in range(1, 11)]
for sch in ("within", "full", "runs"):
    os_ = [o for o in new if o.startswith(sch)]
    for v in ("v6f", "v6.1", "fallback", "loss"):
        M = np.array([new[o][v] for o in os_]); pp = M.mean(0)  # per person, mean over orders
        bo = [np.mean(rng.choice(M.mean(1), len(os_))) for _ in range(4000)]                  # the table's interval: orders only
        bp = [pp[rng.integers(0, 10, 10)].mean() for _ in range(4000)]                         # people only
        b2 = [M[rng.integers(0, len(os_), len(os_))][:, rng.integers(0, 10, 10)].mean() for _ in range(4000)]  # both
        q = lambda x: "[%+.2f, %+.2f]" % tuple(np.percentile(x, [10, 90]))
        q95 = lambda x: "[%+.2f, %+.2f]" % tuple(np.percentile(x, [2.5, 97.5]))
        print(f"new {sch:6s} {v:8s} n_orders {len(os_):2d} mean {M.mean():+.2f} | 80% orders-only {q(bo)} people-only {q(bp)} both {q(b2)} | 95% both {q95(b2)} | "
              f"sd over people {pp.std(ddof=1):.2f} | per person " + " ".join(f"{v_:+.1f}" for v_ in pp))
print("new-person absolute v5.1b (recording order) per person:", " ".join(f"{a:.1f}" for a in new[sorted(new)[0]]["abs"]))
for sch in ("within", "full", "runs"):
    os_ = [o for o in ret if o.startswith(sch)]
    for v in ("ER", "ERF", "F"):
        M = np.array([ret[o][v] for o in os_])
        print(f"ret {sch:6s} {v:4s} n_orders {len(os_)} P04 {M[:, 0].mean():+.2f} (sd over orders {M[:, 0].std(ddof=1):.2f}) P10 {M[:, 1].mean():+.2f} ({M[:, 1].std(ddof=1):.2f})")
    print(f"   routed v5.1b absolute: P04 {np.mean([ret[o]['abs'][0] for o in os_]):.1f} P10 {np.mean([ret[o]['abs'][1] for o in os_]):.1f}")
# sealed-population weightings (106 sessions: 60 new, 6 P04/P10, 40 single-session returning)
W_NEW, W_RET = 60 / 106, 46 / 106
print("\nsealed-weighted face value (no transfer discount): w = weight of P04's response among returning sessions")
for cand, nv, rv in (("v6.2", "v6.1", "ERF"), ("v6f+ERF", "v6f", "ERF"), ("v6.2-safe", "v6.1", "ER"), ("v6f+ER", "v6f", "ER"), ("v6.1", "v6.1", None), ("v6f", "v6f", None)):
    for sch in ("within", "full", "runs"):
        n = np.mean([new[o][nv].mean() for o in new if o.startswith(sch)])
        r4 = np.mean([ret[o][rv][0] for o in ret if o.startswith(sch)]) if rv else 0.0
        r10 = np.mean([ret[o][rv][1] for o in ret if o.startswith(sch)]) if rv else 0.0
        out = []
        for w in (0.5, 0.22, 3 / 46, 0.0):
            out.append(f"w={w:.2f}: {W_NEW * n + W_RET * (w * r4 + (1 - w) * r10):+.2f}")
        print(f"  {cand:9s} {sch:6s} new {n:+.2f} P04 {r4:+.2f} P10 {r10:+.2f} | dev J {0.5 * n + 0.25 * (r4 + r10):+.2f} | sealed-weighted " + "  ".join(out))
