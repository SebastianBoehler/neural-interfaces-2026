"""SOFT COUNT FILTER (ledger s251 (0) and (1)), new people, on runs/chainout/s242r15_<order>.npz V6Fr (six bases).
(0) hard diagnostic: s250's G1 online groups (runs/fingerprint/groups_G1.npz, full orders only) through
    jointcount.filt_groups (cap 12, tau 2, floor 0.5), against raw.
(1) SCF: per session in delivery order, prior_c ~ max(CAP - k_c, 0) + FLOOR, k_c = sum_{j<t} w'_tj q_j(c) (q_j the
    filter's own emitted output, w' from fpw.py), likelihood P_t^TAU; output ~ prior x likelihood. Causal, per session.
    Settings: kernel g (w' = g w) and top-k (w' = w on the k smallest relative distances).
    GSCF: SCF before the s228c latch, latchJ (J-wprior/urn) from it on; scored against latchJ (the current emission).
Nested: per held-out person the setting with the best mean GSCF - latchJ of the nine others over all orders (schemes
equal). Intervals resample people and orders (orders within scheme). Writes runs/softfp/softcount.npz.
Usage: [SRC=runs/chainout/s242r15_{order}.npz] python softcount.py"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, HERE + "/../rundetect")
import fpw  # noqa: E402
import package as K  # noqa: E402  (latch, gate, J, C)
C, J = K.C, K.J; SC = C.A.SC
TAU, FLOOR, CAP = 2.0, 0.5, 11.0
SETTINGS = [("g", 0.5), ("g", 1.0), ("g", 2.0), ("k", 5), ("k", 10), ("k", 20)] + ([("g", 0.0)] if os.environ.get("ZERO") else [])  # ZERO: g 0 must equal raw
SET = os.environ.get("SET", "v1")  # v2 (s251 amendment 1b): + normalised soft count m<M> and the gamma curve's shape
NESTS = {"nested": SETTINGS}
if SET == "v2":
    MS = [("m", v) for v in (5, 10, 15, 20, 30)]; NESTS = {"nested (1b, M)": MS, "nested (all 11)": SETTINGS + MS}
    SETTINGS = SETTINGS + MS + [("g", 0.75), ("g", 1.25), ("g", 1.5)]
SRC = os.environ.get("SRC", "runs/chainout/s242r15_{order}.npz")
ORDERS = os.environ.get("ORDERS", ",".join(f"{s}{i}" for s in ("full", "within", "runs") for i in range(4))).split(",")


def wprime(W, st):
    kind, v = st
    if kind in ("g", "m"):
        return (v if kind == "g" else 1.0) * W
    out = np.zeros_like(W); T = len(W)
    for t in range(1, T):
        j = np.argsort(-W[t, :t], kind="stable")[:int(v)]; out[t, j] = W[t, j]  # top weights = smallest r (monotone calibration)
    return out


def scf(Pb, sw, M=None):
    """Pb (nb, N, 3) outputs, sw {session: (rows, W')} -> filtered (nb, N, 3); M: counts scaled to at most M (1b)"""
    out = Pb.copy()
    for s, (rows, Wp) in sw.items():
        Q = np.zeros((Pb.shape[0], len(rows), 3)); L = np.clip(Pb[:, rows], 1e-9, 1) ** TAU; L /= L.sum(-1, keepdims=True)
        for t in range(len(rows)):
            k = np.einsum("j,bjc->bc", Wp[t, :t], Q[:, :t]); k = k if M is None else k * min(1.0, M / max(Wp[t, :t].sum(), 1e-12)); pr = np.maximum(CAP - k, 0) + FLOOR; q = pr * L[:, t]; Q[:, t] = q / q.sum(-1, keepdims=True)
        out[:, rows] = Q
    return out


def boot(d, rng, n=4000):
    """d (orders, people) -> 80 % interval of the mean, resampling orders and people"""
    o, p = d.shape; return np.percentile([d[rng.integers(0, o, o)][:, rng.integers(0, p, p)].mean() for _ in range(n)], [10, 90])


def main():
    os.makedirs(C.REPO + "/runs/softfp", exist_ok=True); rng = np.random.default_rng(0); G = {}; R = {}
    g1 = np.load(C.REPO + "/runs/fingerprint/groups_G1.npz")
    print(f"SOFTCOUNT (s251): src {SRC}, TAU {TAU}, FLOOR {FLOOR}, CAP {CAP}; V6Fr, new people, mean over bases", flush=True)
    for order in ORDERS:
        sc = order.rstrip("0123456789"); g = C.PERMS[f"{sc}|{order[len(sc):]}|{C.p0}"]; rk = np.empty(C.N, int); rk[g] = np.arange(C.N)
        z = np.load(C.REPO + "/" + SRC.format(order=order)); bases = sorted({k.split("|")[0] for k in z.files if k.endswith("|V6Fr")})
        Pb = np.stack([z[f"{bn}|V6Fr"].astype(np.float64) for bn in bases]); assert Pb.shape[1] == C.N
        lat = K.latch(order); hc = J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
        FJ = np.stack([J.filt_joint(P, hc, "urn") for P in Pb]); LJ = np.stack([K.gate(P, F, order, lat) for P, F in zip(Pb, FJ)])
        sc_ = lambda X: np.mean([SC.person(x, C.p0) for x in X], 0)  # noqa: E731
        raw, lj = sc_(Pb), sc_(LJ); R[order, "latchJ-raw"] = lj - raw
        if order in g1.files:
            lab = np.char.add(C.SES.astype(str), np.char.add("|", g1[order].astype(str)))
            R[order, "G1-raw"] = sc_([J.filt_groups(P, rk, lab, lambda r: 12) for P in Pb]) - raw
        W = fpw.all_weights(order)
        for st in SETTINGS:
            sw = {s: (rows, wprime(Wm, st)) for s, (rows, Wm) in W.items()}; S = scf(Pb, sw, st[1] if st[0] == "m" else None)
            GS = np.stack([K.gate(s_, l_, order, lat) for s_, l_ in zip(S, LJ)])  # SCF before the latch, latchJ from it on
            R[order, f"SCF{st}-raw"] = sc_(S) - raw; G[order, st] = R[order, f"GSCF{st}-latchJ"] = sc_(GS) - lj
        print(f"  {order:8s} latched {sum(w is not None for w in lat.values())}/{len(lat)} | latchJ-raw {R[order, 'latchJ-raw'].mean():+.2f}"
              + (f" | G1-raw {R[order, 'G1-raw'].mean():+.2f}" if (order, "G1-raw") in R else "")
              + " | " + " ".join(f"{st[0]}{st[1]:g}: SCF {R[order, f'SCF{st}-raw'].mean():+.2f} GSCF {G[order, st].mean():+.2f}" for st in SETTINGS), flush=True)
    # nested choice per held-out person
    NN = {}
    for nname, cand in NESTS.items():
        sel, N = {}, {o: np.zeros(10) for o in ORDERS}
        for i, h in enumerate(C.PEOPLE):
            crit = {st: np.mean([np.mean([np.delete(G[o, st], i).mean() for o in ORDERS if o.startswith(s)]) for s in ("full", "within", "runs")]) for st in cand}
            sel[h] = max(crit, key=crit.get)
            for o in ORDERS:
                N[o][i] = G[o, sel[h]][i]
        NN[nname] = N; print(f"{nname} choice: " + " ".join(f"{h} {k}{v:g}" for h, (k, v) in sel.items()))
    print("SUMMARY: mean over orders per scheme [80 % interval, people and orders resampled]" + (" (smoke: fewer orders)" if len(ORDERS) < 12 else "") + " | people up")
    rows = [(f"GSCF {nn} - latchJ", lambda o, nn=nn: NN[nn][o]) for nn in NN] + [(f"GSCF {k}{v:g} - latchJ", lambda o, st=(k, v): G[o, st]) for k, v in SETTINGS] + \
        [(f"SCF {k}{v:g} - raw", lambda o, st=(k, v): R[o, f"SCF{st}-raw"]) for k, v in SETTINGS] + [("latchJ - raw", lambda o: R[o, "latchJ-raw"]), ("G1 hard - raw", lambda o: R.get((o, "G1-raw")))]
    for name, f in rows:
        cells = []
        for s in ("full", "within", "runs"):
            d = [f(o) for o in ORDERS if o.startswith(s)]
            if not d or any(x is None for x in d):
                continue
            d = np.array(d); b = boot(d, rng); cells.append(f"{s} {d.mean():+.2f} [{b[0]:+.2f}, {b[1]:+.2f}] {int((d.mean(0) > 0).sum())}/10")
        print(f"  {name:22s} " + " | ".join(cells), flush=True)
    np.savez(C.REPO + f"/runs/softfp/softcount{'' if SET == 'v1' else '_' + SET}.npz", **{f"{o}|{k}": v for (o, k), v in R.items()}, **{f"{o}|{nn}": v for nn, N in NN.items() for o, v in N.items()})


if __name__ == "__main__":
    main()
