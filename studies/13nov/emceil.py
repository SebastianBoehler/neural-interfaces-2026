"""EMCEIL (ledger s180; s174 item 2): where the stage loss under shuffled delivery comes from, and how much an
order-free estimate could recover. SessionEM already refits an EM fixed point over every delivered window at each
step and TSEM rebuilds its template from all earlier windows, so the order enters through (a) the rebalancer's running
window (W 30), (b) the prefix itself, (c) TSEM labels fixed when each window was emitted. Arms (new people, members as
recorded, stages in the delivery order, mean over the six bases, per-person score):
  C0     the shipped stages (= gm.pipe_g)
  Winf   rebalancer over all earlier windows of the session (W = inf): causal and order-free over the prefix
  EMc    SessionEM ceiling: mu and covariance fitted on the whole session (non-causal), causal exponent schedule
  TSc    TSEM ceiling: template from the whole session (labels = SessionEM output argmax, self included), same schedule
  ALLc   Winf + EMc + TSc
Usage: [ORDERS=rec:0,within:0,full:0,runs:0] python studies/13nov/emceil.py"""
import os, sys
import numpy as np
os.environ.setdefault("SCHEMES", "within,full,runs"); os.environ.setdefault("PERMS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shuffle as S  # noqa: E402
A, SC, gm, p0 = S.A, S.SC, S.gm, S.p0; Lb, M, SE = gm.L, gm.M, gm.SE
ORDERS = [tuple(x.split(":")) for x in os.environ.get("ORDERS", "rec:0,within:0,full:0,runs:0").split(",")]


def em_ceil(b, g):
    F, sess = Lb.Z[f"GE|{p0}"], Lb.Z[f"sess|{p0}"]; out = b.copy()
    for s in dict.fromkeys(sess):
        r = np.flatnonzero(sess == s); X = F[r]; m, sd = X.mean(0), np.maximum(X.std(0), SE.SD_FLOOR); Zs = (X - m) / sd; Q = b[r]
        try:
            for _ in range(2):
                mu = (Q / (Q.sum(0, keepdims=True) + 20)).T @ Zs; C = np.cov((Zs - Q @ mu).T)
                Ci = np.linalg.inv(0.3 * C + 0.7 * np.eye(len(C)) * np.trace(C) / len(C))
                Q = b[r] * SE._softmax_lda(Zs, mu, Ci) ** 0.5; Q = Q / Q.sum(1, keepdims=True)
        except np.linalg.LinAlgError:
            continue
        t = np.arange(len(r)); e = 0.5 * t / (t + 60) * (1 + gm.BEM * (M.h(t, 30) * g[r]))
        q = b[r] * SE._softmax_lda(Zs, mu, Ci) ** e[:, None]; q = q / q.sum(1, keepdims=True); ok = t >= gm.NMIN; out[r[ok]] = q[ok]
    return out


def ts_ceil(P, g, rho=0.7, m0=20):
    X, sess = Lb.Z[f"pc128|{p0}"], Lb.Z[f"sess|{p0}"]; out = P.copy()
    for s in dict.fromkeys(sess):
        r = np.flatnonzero(sess == s); lab, conf = P[r].argmax(1), P[r].max(1); n = len(r); q = max(2, int(rho * n / 3))
        sel = [np.flatnonzero(lab == k)[np.argsort(-conf[lab == k])[:q]] for k in range(3)]
        if min(len(x) for x in sel) < 2:
            continue
        m, sd = X[r].mean(0), np.maximum(X[r].std(0), 1e-3); Zm = (X[r] - m) / sd
        mu = np.stack([Zm[x].sum(0) / (len(x) + m0) for x in sel]); R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)])
        Cv = np.cov(R.T); Cv = 0.3 * Cv + 0.7 * np.eye(len(Cv)) * np.trace(Cv) / len(Cv); Lk = Lb.G.T._lda(Zm, mu, np.linalg.inv(Cv))
        t = np.arange(n); e = A.LAM * t / (t + 60) * (1 + gm.BTS * (M.h(t, 30) * g[r])); qq = P[r] * Lk ** e[:, None]; qq = qq / qq.sum(1, keepdims=True)
        ok = t >= 15; out[r[ok]] = qq[ok]
    return out


def arms(bn):
    mix = A.mix(bn, p0, A.FAMS)[0]; sess = Lb.Z[f"sess|{p0}"]; res = {}
    for name, W, emc, tsc in (("C0", 30, 0, 0), ("Winf", 10 ** 6, 0, 0), ("EMc", 30, 1, 0), ("TSc", 30, 0, 1), ("ALLc", 10 ** 6, 1, 1)):
        b = Lb.rebal(mix, sess, W=W); g = gm.gate(b, p0); e = em_ceil(b, g) if emc else gm.em_g(b, p0, g); t = ts_ceil(e, g) if tsc else gm.tsem_g(e, p0, g)
        res[name] = SC.person(t, p0)
    assert np.allclose(res["C0"], SC.person(gm.pipe_g(mix, p0), p0)), "arm C0 is not the shipped stages"
    return res


R = {}
for sc, sd in ORDERS:
    P = {p: (np.arange(S.N[p]) if sc == "rec" else S.perm_for(sc, int(sd))[p]) for p in S.PR}; S.apply(P)
    R[sc + sd] = {bn: arms(bn) for bn in S.BASES}
    print(f"{sc}{sd}: " + " | ".join(f"{k} {np.mean([R[sc + sd][bn][k].mean() for bn in S.BASES]):.2f}" for k in R[sc + sd][S.BASES[0]]), flush=True)
S.apply({p: np.arange(S.N[p]) for p in S.PR}); ref = "rec0"; rng = np.random.default_rng(0)
for o in R:
    for k in R[o][S.BASES[0]]:
        d = np.mean([R[o][bn][k] - R[ref][bn]["C0"] for bn in S.BASES], 0); bt = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)], [10, 90])
        print(f"{o:8s} {k:5s} against shipped stages in recording order: new {d.mean():+.2f} [{bt[0]:+.2f}, {bt[1]:+.2f}] without P10 {np.delete(d, 9).mean():+.2f}", flush=True)
np.savez(S.REPO + "/runs/emceil.npz", **{f"{o}|{bn}|{k}": v for o, r in R.items() for bn, rr in r.items() for k, v in rr.items()})
