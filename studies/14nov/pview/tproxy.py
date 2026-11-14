"""PV6: the TSEM personal template prior (PV2's arm T) on a within-session proxy for all ten people.
Each public session is split by task: the two acquisition runs (one Graz, one BrainHero) play the person's public
session (labels used offline: class means of their core1 PCs, z-scored over those rows); the four online runs play the
later session and get the prior (kappa pseudo-windows, denominator included) in TSEM's class means. Chain: the
new-people tail of v5.1b (mix -> rebal -> gated SessionEM -> gated TSEM, gm.pipe_g) on all_loso, members as recorded
(shuffle.py's limit), stages in the delivery order. Scored on online-run rows only (cells person x session x
interface); acquisition rows must be unchanged (asserted). Optimistic against the real case: same session, so no
between-session shift of the templates (PV1: personal pc128 templates 77 within a session vs 67 across sessions).
Usage: [BASES=..] [ORDERS=rec,within0,..] [KAPS=20,40] python studies/14nov/pview/tproxy.py"""
import os, sys
import numpy as np
os.environ.setdefault("SCHEMES", "within,full,runs"); os.environ.setdefault("PERMS", "4")
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.abspath(__file__ + "/../../../13nov")]
import shuffle as S  # noqa: E402
import data as D  # noqa: E402
A, SC, gm, p0 = S.A, S.SC, S.gm, S.p0; G, Lz = gm.L.G, gm.L.Z
BASES = os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(",")
ORDERS = os.environ.get("ORDERS", "rec,within0,within1,within2,within3,full0,full1,runs0,runs1").split(",")
KAPS = [float(k) for k in os.environ.get("KAPS", "20,40").split(",")]
TASK = np.load(D.REPO + "/data/proteus_windows.npz")["task"].astype(str); ONL = np.char.startswith(TASK, "Online")
_lik0 = G.tsem_lik


def prior_rows(idx):
    """per row (store layout, global index idx): prior means (3, 128) from the session's acquisition rows, and kp mask."""
    PM, ON = np.zeros((len(idx), 3, 128)), ONL[idx]
    for s in np.unique(D.SESS):
        acq = np.flatnonzero((D.SESS == s) & ~ONL); X = D.PC[acq]; X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-3)
        pm = np.stack([X[D.Y[acq] == k].mean(0) for k in range(3)]); PM[D.SESS[idx] == s] = pm
    return PM, ON.astype(float)


def lik_prior(PM, KP):  # = grid.tsem_lik with kp pseudo-windows of PM in the class means (as chain.tsem_lik_prior)
    T_ = G.T

    def f(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
        Lk = np.full_like(P, np.nan); tt = np.zeros(len(P))
        for s in np.unique(sess):
            r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1)
            for t in range(nmin, len(r)):
                A_, fx = X[r[:t]], X[r[t]]; m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3); Zm, z = (A_ - m) / sd, (fx - m) / sd
                q = max(2, int(rho * t / 3)); sel = [np.flatnonzero(lab[:t] == k)[np.argsort(-conf[np.flatnonzero(lab[:t] == k)])[:q]] for k in range(3)]
                if min(len(x) for x in sel) < 2:
                    continue
                kp = KP[r[t]]; mu = np.stack([(Zm[x].sum(0) + kp * PM[r[t], k]) / (len(x) + m0 + kp) for k, x in enumerate(sel)])
                R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)]); Cv = np.cov(R.T)
                Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
                Lk[r[t]] = T_._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
        return Lk, tt
    return f


def ba_online(P, idx):
    on = ONL[idx]; per = D.PER[idx]
    return np.array([D.ba(idx[on & (per == q)], P[on & (per == q)].argmax(1)) for q in D.PEOPLE])


R = {}
for order in ORDERS:
    P = {q: np.arange(S.N[q]) for q in S.PR} if order == "rec" else S.perm_for(order.rstrip("0123456789"), int(order[len(order.rstrip("0123456789")):]))
    S.apply(P); idx = Lz[f"idx|{p0}"].astype(int); assert (idx == np.arange(D.N)[P[p0]]).all(), "store layout is not the permutation"
    PM, ON = prior_rows(idx)
    for bn in BASES:
        mix = A.mix(bn, p0, A.FAMS)[0]; G.tsem_lik = _lik0; base = gm.pipe_g(mix, p0); R[order, bn, "base"] = ba_online(base, idx)
        for k in KAPS:
            G.tsem_lik = lik_prior(PM, k * ON)
            try:
                out = gm.pipe_g(mix, p0)
            finally:
                G.tsem_lik = _lik0
            assert np.array_equal(out[ON == 0], base[ON == 0]), "the prior reached acquisition rows"
            R[order, bn, f"T{k:g}"] = ba_online(out, idx)
        print(f"{order} {bn}: online base {R[order, bn, 'base'].mean():.2f} | " + " ".join(f"T{k:g} {(R[order, bn, f'T{k:g}'] - R[order, bn, 'base']).mean():+.2f}" for k in KAPS), flush=True)
S.apply({q: np.arange(S.N[q]) for q in S.PR})
os.makedirs(D.OUT, exist_ok=True); np.savez(f"{D.OUT}/tproxy.npz", **{f"{o}|{b}|{a}": v for (o, b, a), v in R.items()})
print("\nPV6: change on online runs against the unprimed tail, six-base mean; per person, then mean over orders")
for k in KAPS:
    d = np.array([[R[o, b, f"T{k:g}"] - R[o, b, "base"] for b in BASES] for o in ORDERS]).mean(1)  # (orders, people)
    print(f"T{k:g}: mean {d.mean():+.2f}; people > 0 {(d.mean(0) > 0).sum()}/10; " + " ".join(f"{q} {v:+.2f}" for q, v in zip(D.PEOPLE, d.mean(0))))
    for sc in ("rec", "within", "full", "runs"):
        oo = [i for i, o in enumerate(ORDERS) if o.rstrip("0123456789") == sc]
        if oo:
            m = d[oo].mean(1); print(f"   {sc:6s} {m.mean():+.2f} (orders {', '.join(f'{x:+.2f}' for x in m)})")
