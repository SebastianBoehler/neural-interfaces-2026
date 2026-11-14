"""PACKAGE-TRUE COUNT FILTER (ledger s228c). Per session in delivery order the 'lag' detector (detect.py; thresholds fitted
on the other people, runs/rundetect/decisions.npz) is checked at W in CHECKS and latches at the first pass for good.
The emitted output is the chain posterior before the latch window and J-wprior/urn (jointcount.py) from it on; the urn
and hypothesis state use every window <= t (causal), only the emission switches. Arms per order:
  J        always on (reference)          latchJ    (a) detector-gated
  latchJon (b) gated, urn on online blocks only (acquisition blocks unfiltered)
  true     true runs, always on           trueon    true runs, online runs only (true caps)
(c) gain by delivery position: J and true applied only to windows at session positions in each bin (within orders).
Input: per-window outputs, SRC pattern (default runs/chainout/{order}.npz: <base>|<arm> -> (N, 3), recording layout).
Usage: [SRC=...] [ARMS=P0r,V6Fr] python package.py order [order ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import detect as D  # noqa: E402
import jointcount as J  # noqa: E402

CHECKS = (40, 50, 60, 80, 100, 150); BINS = (0, 40, 80, 120, 160, 10 ** 6)
SRC = os.environ.get("SRC", "runs/chainout/{order}.npz"); ARMS = os.environ.get("ARMS", "P0r,V6Fr").split(",")
THR = np.load(C.REPO + "/runs/rundetect/decisions.npz"); WS = list(THR["W"]); X, LAM = C.projected()


def latch(order):
    """-> {session: latch window or None}"""
    out = {}
    for s in C.SESSIONS:
        rows = C.order_rows(order, s); p = C.PER[rows[0]]; v = D.stat(X, rows, "lag", [w for w in CHECKS if w <= len(rows)])
        hit = [w for w, x in zip(CHECKS, v) if x < THR[f"lag|{p}"][WS.index(w)]]; out[s] = hit[0] if hit else None
    return out


def gate(P, F, order, lat):
    out = P.copy()
    for s, w in lat.items():
        if w is not None:
            r = C.order_rows(order, s)[w:]; out[r] = F[r]
    return out


def posmask(order, lo, hi):
    m = np.zeros(C.N, bool)
    for s in C.SESSIONS:
        m[C.order_rows(order, s)[lo:hi]] = True
    return m


def main():
    rng = np.random.default_rng(0); res = {}
    print(f"PACKAGE COUNT FILTER (s228c): src {SRC}, TAU {J.TAU}, checks {CHECKS}; change against filter off, new people, mean over bases")
    for order in sys.argv[1:]:
        sc = order.rstrip("0123456789"); g = C.PERMS[f"{sc}|{order[len(sc):]}|{C.p0}"]; rk = np.empty(C.N, int); rk[g] = np.arange(C.N)
        z = np.load(C.REPO + "/" + SRC.format(order=order)); bases = sorted({k.split("|")[0] for k in z.files if k.split("|")[1] in ARMS})
        lat = latch(order); hc = J.hyp_cache(order, "wprior", X, LAM); hco = J.hyp_cache(order, "wprior", X, LAM, online_only=True)
        online = np.array(["Online" in u.split("|")[1] for u in C.RUN]); masks = [posmask(order, a, b) for a, b in zip(BINS[:-1], BINS[1:])]
        nl = sum(w is not None for w in lat.values()); wl = [w for w in lat.values() if w is not None]
        for arm in ARMS:
            d = {}
            for bn in bases:
                P = z[f"{bn}|{arm}"].astype(np.float64); base = C.A.SC.person(P, C.p0); FJ = J.filt_joint(P, hc, "urn"); FJo = J.filt_joint(P, hco, "urn")
                FT = J.filt_groups(P, rk, C.RUNI, lambda r: J.TRUE_CAP[C.RUN[r[0]]])
                FTo = P.copy(); lab = np.where(online, C.RUNI, -1); FTo[online] = J.filt_groups(P, rk, lab, lambda r: J.TRUE_CAP[C.RUN[r[0]]] if lab[r[0]] >= 0 else 0)[online]
                outs = {"J": FJ, "latchJ": gate(P, FJ, order, lat), "latchJon": gate(P, FJo, order, lat), "true": FT, "trueon": FTo}
                for i, m in enumerate(masks):
                    outs[f"J@{i}"] = np.where(m[:, None], FJ, P); outs[f"true@{i}"] = np.where(m[:, None], FT, P)
                for k, o in outs.items():
                    d.setdefault(k, []).append(C.A.SC.person(o, C.p0) - base)
            for k, v in d.items():
                res[order, arm, k] = np.mean(v, 0)
            print(f"  {order:8s} {arm:4s} latched {nl}/14 at W {np.median(wl) if wl else '-'} | " + " ".join(f"{k} {res[order, arm, k].mean():+.2f}" for k in ("J", "latchJ", "latchJon", "true", "trueon")), flush=True)
    print("SUMMARY per scheme: mean over orders, 80 % person bootstrap of the order-mean per-person change")
    for sc in ("within", "runs", "full"):
        os_ = [o for o in sys.argv[1:] if o.rstrip("0123456789") == sc]
        if not os_:
            continue
        for arm in ARMS:
            cells = []
            for k in ("J", "latchJ", "latchJon", "true", "trueon") + tuple(f"{f}@{i}" for f in ("true", "J") for i in range(len(BINS) - 1)):
                d = np.mean([res[o, arm, k] for o in os_], 0); bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90])
                cells.append(f"{k} {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}]")
            print(f"  {sc:6s} {arm:4s} ({len(os_)} orders) " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
