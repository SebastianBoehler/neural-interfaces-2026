"""Stacked scoring of the causal C10 member (ledger s256), new people, the 20 orders (within0-11, full0-3, runs0-3).
Net = seed mean of runs/tttssl/causal/c10-s<seed>-g<group>.npz (arm A adapted, arm B unadapted). Per order and base:
out = chain x net^W renormalised on every row from session delivery position START on (rows before unchanged), chain =
runs/chainout/s242r15_<order>.npz <base>|V6Fr; then the detector-gated count filter exactly as studies/14nov/stack/score.py
(K.gate(P, K.J.filt_joint(P, hc, "urn"), order, lat)). Arms: cand (r15 + filter), A+filter, B+filter (and raw forms).
Changes per order x person (mean over bases); intervals resample people and orders within scheme, schemes equal weight
(4000 draws, 80 %); people up = person means of the scheme-equal mean. No returning-people form: absent.
Zero-change checks: W = 0 returns the chain (raw and filtered); orders == the pinned perm file; arm A = arm B before START.
Usage: python studies/14nov/tttssl/stack.py [W=0.5] > evidence/tttssl_14nov/stack.txt"""
import glob, os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5; sys.argv = sys.argv[:1]
sys.path.insert(0, REPO + "/studies/14nov/rundetect")
import package as K  # noqa: E402
START = 20; SCHEMES = ("within", "full", "runs")
ORDERS = [f"{s}{i}" for s, n in (("within", 12), ("full", 4), ("runs", 4)) for i in range(n)]
rng = np.random.default_rng(0)


def seed_mean():
    fs = sorted(glob.glob(f"{REPO}/runs/tttssl/causal/c10-s*-g*.npz")); assert len(fs) == 30, f"{len(fs)} files, need 30 (6 seeds x 5 groups)"
    acc = {}
    for f in fs:
        z = np.load(f)
        for k in z.files:
            acc.setdefault(k, {})[os.path.basename(f).split("-")[1]] = z[k]
    out = {}
    for k, v in acc.items():
        assert len(v) == 6, (k, sorted(v)); out[k] = np.mean(list(v.values()), 0)
    return out


def fuse(c, q, rows_late, w):
    F = c.copy(); r = rows_late; F[r] = c[r] * np.clip(q[r], 1e-9, 1) ** w; F[r] /= F[r].sum(1, keepdims=True); return F


def main():
    net = seed_mean(); R = {}; chk = []
    for order in ORDERS:
        lat = K.latch(order); hc = K.J.hyp_cache(order, "wprior", K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
        late = np.concatenate([K.C.order_rows(order, s, K.PROTO)[START:] for s in K.SESS]); early = np.concatenate([K.C.order_rows(order, s, K.PROTO)[:START] for s in K.SESS])
        assert np.abs(net[f"A|{order}"][early] - net[f"B|{order}"][early]).max() < 1e-4
        z = np.load(f"{REPO}/runs/chainout/s242r15_{order}.npz"); bases = sorted({k.split("|")[0] for k in z.files if k.endswith("|V6Fr")}); assert len(bases) == 6
        S = {k: [] for k in ("cand", "cand_raw", "A", "B", "A_raw", "B_raw")}
        for bn in bases:
            P = np.full((K.C.N, 3), 1 / 3); P[K.ROWS] = z[f"{bn}|V6Fr"].astype(np.float64); sc = lambda o: K.C.A.SC.person(o[K.ROWS], K.PROTO)  # noqa: E731
            filt = lambda Q: K.gate(Q, K.J.filt_joint(Q, hc, "urn"), order, lat)  # noqa: E731
            S["cand_raw"].append(sc(P)); S["cand"].append(sc(filt(P)))
            if order == "within0" and bn == bases[0]:  # W = 0 is the chain
                P0 = fuse(P, net["A|within0"], late, 0.0); assert np.abs(P0 - P).max() < 1e-12 and np.abs(sc(filt(P0)) - sc(filt(P))).max() < 1e-12; chk.append("w0 ok")
            for a in "AB":
                Q = np.full((K.C.N, 3), 1 / 3); Q[K.ROWS] = net[f"{a}|{order}"]; Fq = fuse(P, Q, late, W)
                S[a + "_raw"].append(sc(Fq)); S[a].append(sc(filt(Fq)))
        m = {k: np.mean(v, 0) for k, v in S.items()}
        for k in ("A", "B"):
            R[order, f"{k}+f - cand+f"] = m[k] - m["cand"]; R[order, f"{k} raw - cand raw"] = m[k + "_raw"] - m["cand_raw"]
        R[order, "A - B (filtered)"] = m["A"] - m["B"]; R[order, "A - B (raw)"] = m["A_raw"] - m["B_raw"]; R[order, "cand+f - cand raw"] = m["cand"] - m["cand_raw"]
        print(f"  {order:8s} cand {m['cand'].mean():.2f} | " + " ".join(f"{k} {v.mean():+.2f}" for (o, k), v in R.items() if o == order), flush=True)
    keys = list(dict.fromkeys(k for _, k in R)); print(f"\nW {W}, {chk}; new people; change [80 % people x orders] | people up (scheme-equal mean over orders)")
    for k in keys:
        D = {s: np.stack([R[o, k] for o in ORDERS if o.startswith(s)]) for s in SCHEMES}; pt = np.mean([d.mean() for d in D.values()]); bt = []
        for _ in range(4000):
            pp = rng.integers(0, 10, 10); bt.append(np.mean([d[rng.integers(0, len(d), len(d))][:, pp].mean() for d in D.values()]))
        pm = np.mean([d.mean(0) for d in D.values()], 0); lo, hi = np.percentile(bt, [10, 90])
        print(f"  {k:22s} {pt:+.2f} [{lo:+.2f}, {hi:+.2f}] {int((pm > 0).sum())}/10 | " + " ".join(f"{s} {d.mean():+.2f}" for s, d in D.items()) + " | per person " + " ".join(f"{x:+.1f}" for x in pm))


if __name__ == "__main__":
    main()
