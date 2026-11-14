"""PACKAGE-TRUE COUNT FILTER (ledger s228c). Per session in delivery order the 'lag' detector (detect.py; thresholds fitted
on the other people, runs/rundetect/decisions.npz) is checked at W in CHECKS and latches at the first pass for good.
The emitted output is the chain posterior before the latch window and J-wprior/urn (jointcount.py) from it on; the urn
and hypothesis state use every window <= t (causal), only the emission switches. Arms per order:
  J        always on (reference)          latchJ    (a) detector-gated
  latchJon (b) gated, urn on online blocks only (acquisition blocks unfiltered)
  true     true runs, always on           trueon    true runs, online runs only (true caps)
(c) gain by delivery position: J and true applied only to windows at session positions in each bin (within orders).
Input: per-window outputs, SRC pattern (default runs/chainout/{order}.npz: <base>|<arm> -> (N, 3), recording layout).
PROTO=seen_session (s238): returning people; SRC default runs/chainout/ret_{order}.npz with keys <order>|<base>|<arm>,
(n_seen, 3) in the seen_session layout; the filter runs on those sessions with their own delivery orders, the detector
thresholds and design prior still leave the person out; scores per person (P04, P10), no bootstrap.
Usage: [PROTO=all_loso|seen_session] [SRC=...] [ARMS=P0r,V6Fr] python package.py order [order ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import detect as D  # noqa: E402
import jointcount as J  # noqa: E402

CHECKS = (40, 50, 60, 80, 100, 150); BINS = (0, 40, 80, 120, 160, 10 ** 6)
PROTO = os.environ.get("PROTO", "all_loso"); SEEN = PROTO == "seen_session"
SRC = os.environ.get("SRC", "runs/chainout/ret_{order}.npz" if SEEN else "runs/chainout/{order}.npz"); ARMS = os.environ.get("ARMS", "ER,ERF,base" if SEEN else "P0r,V6Fr").split(",")
ROWS = C.A.pl.B[f"idx|{PROTO}"].astype(int); SESS = list(dict.fromkeys(C.SES[ROWS]))  # global rows of the protocol, its sessions
assert (C.A.L.Z[f"y|{PROTO}"] == C.Y[ROWS]).all(), "protocol rows do not map onto the global index"
THR = np.load(C.REPO + "/runs/rundetect/decisions.npz"); WS = list(THR["W"]); X, LAM = C.projected()


def latch(order):
    """-> {session: latch window or None}"""
    out = {}
    for s in SESS:
        rows = C.order_rows(order, s, PROTO); p = C.PER[rows[0]]; v = D.stat(X, rows, "lag", [w for w in CHECKS if w <= len(rows)])
        hit = [w for w, x in zip(CHECKS, v) if x < THR[f"lag|{p}"][WS.index(w)]]; out[s] = hit[0] if hit else None
    return out


def gate(P, F, order, lat):
    out = P.copy()
    for s, w in lat.items():
        if w is not None:
            r = C.order_rows(order, s, PROTO)[w:]; out[r] = F[r]
    return out


def posmask(order, lo, hi):
    m = np.zeros(C.N, bool)
    for s in SESS:
        m[C.order_rows(order, s, PROTO)[lo:hi]] = True
    return m


def main():
    rng = np.random.default_rng(0); res = {}; inP = np.zeros(C.N, bool); inP[ROWS] = True
    online = np.array(["Online" in u.split("|")[1] for u in C.RUN]); score = lambda o: C.A.SC.person(o[ROWS], PROTO)  # noqa: E731
    print(f"PACKAGE COUNT FILTER (s228c/s238): {PROTO}, src {SRC}, TAU {J.TAU}, checks {CHECKS}; change against filter off, mean over bases")
    for order in sys.argv[1:]:
        sc = order.rstrip("0123456789"); g = C.PERMS[f"{sc}|{order[len(sc):]}|{PROTO}"]; rk = np.full(C.N, C.N + 1); rk[g] = np.arange(len(g))
        z = np.load(C.REPO + "/" + SRC.format(order=order)); key = (lambda bn, arm: f"{order}|{bn}|{arm}") if SEEN else (lambda bn, arm: f"{bn}|{arm}")  # noqa: E731
        bases = sorted({k.split("|")[-2] for k in z.files if k.split("|")[-1] in ARMS})
        lat = latch(order); hc = J.hyp_cache(order, "wprior", X, LAM, sessions=SESS, proto=PROTO); hco = J.hyp_cache(order, "wprior", X, LAM, online_only=True, sessions=SESS, proto=PROTO)
        masks = [posmask(order, a, b) for a, b in zip(BINS[:-1], BINS[1:])]; nl = sum(w is not None for w in lat.values()); wl = [w for w in lat.values() if w is not None]
        labT, labO = np.where(inP, C.RUNI, -1), np.where(inP & online, C.RUNI, -1)
        for arm in ARMS:
            d = {}
            for bn in bases:
                P = np.full((C.N, 3), 1 / 3); P[ROWS] = z[key(bn, arm)].astype(np.float64); base = score(P)
                FJ = J.filt_joint(P, hc, "urn"); FJo = J.filt_joint(P, hco, "urn")
                FT = J.filt_groups(P, rk, labT, lambda r: J.TRUE_CAP[C.RUN[r[0]]] if labT[r[0]] >= 0 else 0)
                FTo = J.filt_groups(P, rk, labO, lambda r: J.TRUE_CAP[C.RUN[r[0]]] if labO[r[0]] >= 0 else 0); FTo[~online] = P[~online]
                outs = {"J": FJ, "latchJ": gate(P, FJ, order, lat), "latchJon": gate(P, FJo, order, lat), "true": FT, "trueon": FTo}
                for i, m in enumerate(masks):
                    outs[f"J@{i}"] = np.where(m[:, None], FJ, P); outs[f"true@{i}"] = np.where(m[:, None], FT, P)
                for k, o in outs.items():
                    d.setdefault(k, []).append(score(o) - base)
            for k, v in d.items():
                res[order, arm, k] = np.mean(v, 0)
            print(f"  {order:8s} {arm:4s} latched {nl}/{len(SESS)} at W {np.median(wl) if wl else '-'} | " + " ".join(f"{k} {res[order, arm, k].mean():+.2f}" for k in ("J", "latchJ", "latchJon", "true", "trueon")), flush=True)
    who = list(dict.fromkeys(C.A.L.Z[f"person|{PROTO}"])) if SEEN else None
    print("SUMMARY per scheme: mean over orders" + (", per person" if SEEN else ", 80 % person bootstrap of the order-mean per-person change"))
    for sc in ("within", "runs", "full"):
        os_ = [o for o in sys.argv[1:] if o.rstrip("0123456789") == sc]
        if not os_:
            continue
        for arm in ARMS:
            cells = []
            for k in ("J", "latchJ", "latchJon", "true", "trueon") + tuple(f"{f}@{i}" for f in ("true", "J") for i in range(len(BINS) - 1)):
                d = np.mean([res[o, arm, k] for o in os_], 0)
                if SEEN:
                    cells.append(f"{k} {d.mean():+.2f} (" + " ".join(f"{p} {v:+.2f}" for p, v in zip(sorted(who), d)) + ")")
                else:
                    bt = np.percentile([d[rng.integers(0, 10, 10)].mean() for _ in range(4000)], [10, 90]); cells.append(f"{k} {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}]")
            print(f"  {sc:6s} {arm:4s} ({len(os_)} orders) " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
