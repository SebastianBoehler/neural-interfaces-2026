"""Class-count-generic copies of the chain's stages (ledger s227 item 3): rebalancing, gate, gated SessionEM, gated TSEM
and the routed-path slot blend, written against arrays (sess, views) instead of the replay store, for K classes.
Constants are gm's / mech's / grid's (imported, not retyped). The 3-class literals of the originals become K:
rebalancing prior 1/K and its cumsum width, TSEM's seed quota int(rho * t / K) and its class loop.
`python kstages.py` = parity test against gm on the PROTEUS replay (B2_pool and B2_s0, both protocols, recording
order and one full shuffle): every stage must equal gm's to 1e-10, or it fails loudly."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(HERE + "/../../..")
sys.path.insert(0, REPO + "/replay")
import paths  # noqa: E402,F401
import gm  # noqa: E402
import session_em as SE  # noqa: E402
import tsem as T  # noqa: E402
M = gm.M


def rebal(P, sess, alpha=1.15, k=5.0, W=30):
    K = P.shape[1]; prior = np.full(K, 1 / K); out = np.empty_like(P)
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); X = P[r]
        cs = np.vstack([np.zeros((1, K)), np.cumsum(X, 0)]); n = np.arange(len(X)); lo = np.maximum(0, n - W)
        m = (k * prior + cs[n] - cs[lo]) / (k + (n - lo))[:, None]
        q = X / m ** alpha; out[r] = q / q.sum(1, keepdims=True)
    return out


def gate(b, sess, lo=gm.LO, hi=gm.HI):
    s = np.sort(b, 1); v = s[:, -1] - s[:, -2]; g = np.ones(len(b))
    for x in dict.fromkeys(sess):
        r = np.flatnonzero(sess == x); cs = np.concatenate([[0.0], np.cumsum(v[r])]); t = np.arange(len(r))
        with np.errstate(divide="ignore", invalid="ignore"):
            gg = np.clip((hi - cs[t] / np.maximum(t, 1)) / (hi - lo), 0.0, 1.0)
        g[r] = np.where(t == 0, 1.0, gg)
    return g


def em_g(b, sess, F, g, bem=gm.BEM, nmin=gm.NMIN):
    out = np.empty_like(b)
    for s in dict.fromkeys(sess):
        e = gm.EMg(bem, lam=0.5, n0=60, m0=20, shrink=0.7, nmin=nmin)  # EMg / SessionEM are K-generic as written
        for r in np.flatnonzero(sess == s):
            e.g = g[r]; out[r] = e.step(F[r], b[r])
    return out


def tsem_lik(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
    K = P.shape[1]; Lk = np.full_like(P, np.nan); tt = np.zeros(len(P))
    for s in np.unique(sess):
        r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1)
        for t in range(nmin, len(r)):
            A_, f = X[r[:t]], X[r[t]]
            m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3); Zm, z = (A_ - m) / sd, (f - m) / sd
            q = max(2, int(rho * t / K)); sel = []
            for k in range(K):
                ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
            if min(len(x) for x in sel) < 2:
                continue
            mu = np.stack([Zm[x].sum(0) / (len(x) + m0) for x in sel])
            R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)])
            Cv = np.cov(R.T); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
            Lk[r[t]] = T._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
    return Lk, tt


def tsem_g(P, sess, X, g, lam=0.15, bts=gm.BTS):
    Lk, tt = tsem_lik(P, sess, X); out = P.copy(); m = ~np.isnan(Lk[:, 0])
    e = lam * tt[m] / (tt[m] + 60) * (1 + bts * (M.h(tt[m], 30) * g[m]))
    q = P[m] * Lk[m] ** e[:, None]; out[m] = q / q.sum(1, keepdims=True); return out


def chain(mix, sess, F, X, W=30):
    """-> dict(member, rebal, sessionem, tsem) for one or more sessions (rows in delivery order)."""
    b = rebal(mix, sess, W=W); g = gate(b, sess); e = em_g(b, sess, F, g); t = tsem_g(e, sess, X, g)
    return {"member": mix, "rebal": b, "sessionem": e, "tsem": t}


def slot_blend(mix, H, r, s):
    """Routed path stand-in: the personal head H at strength s where the identity ramp r > 0 (s = the head's share)."""
    w = (s * r)[:, None]; return (1 - w) * mix + w * H


def parity():
    import env  # noqa: F401  (loads the dev replay store)
    A = env.A; Z = A.L.Z; n = 0
    for bn in ("B2_pool", "B2_s0"):
        for p in A.PROTOS:
            mix0 = A.mix(bn, p, A.FAMS)[0]; sess0 = Z[f"sess|{p}"]
            rng = np.random.default_rng(0); perm = np.concatenate([r[rng.permutation(len(r))] for r in
                                                                 (np.flatnonzero(sess0 == s) for s in dict.fromkeys(sess0))])
            for tag, o in (("rec", np.arange(len(sess0))), ("full", perm)):
                key = f"kpar{tag}"
                for v in ("sess", "GE", "pc128"):
                    Z[f"{v}|{key}"] = Z[f"{v}|{p}"][o]
                mix, sess, F, X = mix0[o], Z[f"sess|{key}"], Z[f"GE|{key}"], Z[f"pc128|{key}"]
                b0 = M.rebal(mix, key, None); g0 = gm.gate(b0, key); e0 = gm.em_g(b0, key, g0); t0 = gm.tsem_g(e0, key, g0)
                st = chain(mix, sess, F, X)
                for nm, ref in (("rebal", b0), ("sessionem", e0), ("tsem", t0)):
                    d = np.abs(st[nm] - ref).max(); assert d < 1e-10, f"{bn} {p} {tag} {nm}: max diff {d:.2e} against gm"
                assert np.abs(st["tsem"] - gm.pipe_g(mix, key)).max() < 1e-10, "gm.pipe_g differs from gm's own stages"
                assert np.abs(gate(st["rebal"], sess) - g0).max() == 0
                n += 1
    print(f"PARITY OK: kstages == gm on {n} (base, protocol, order) cells at K = 3 (rebal, gate, SessionEM, TSEM)")


if __name__ == "__main__":
    parity()
