"""STAGEHOOKS (ledger s219): stage variants installed into the replay's stage code by shufcombo, kept out of shufcombo.py.
tsem_target(beta, ...): TSEM's covariance target. grid.tsem_lik shrinks the seed residual covariance 0.7 towards a scaled
identity. Here the target is (1 - beta) * scaled identity + beta * Sigma_pop, where Sigma_pop is the population
within-class, within-session covariance of the session-standardised pc128 view (class means removed per session), estimated
per held-out person on the OTHER people's recording-order rows (labels used offline), scaled to the identity's trace. It
needs no run id: the between-run offsets of the training people are inside Sigma_pop (panel BCI s2.2)."""
import numpy as np


def sigma_pop(V, y, ss, pp):
    """per person: the within-class, within-session covariance of the other people's session-standardised rows."""
    Zs = V.astype(np.float64).copy()
    for g in np.unique(ss):
        m = ss == g; Zs[m] = (Zs[m] - Zs[m].mean(0)) / np.maximum(Zs[m].std(0), 1e-3)
        for c in range(3):
            k = m & (y == c); Zs[k] -= Zs[k].mean(0)
    return {q: np.cov(Zs[pp != q].T) for q in np.unique(pp)}


def tsem_target(beta, G, T, Zstore, p0, SIG):
    """replace G.tsem_lik by a copy whose shrinkage target mixes identity and Sigma_pop of the row's person."""
    def tsem_lik(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
        L = np.full_like(P, np.nan); tt = np.zeros(len(P)); per = Zstore[f"person|{p0}"]
        if len(per) != len(P):  # another protocol (seen_session): the stock stage
            return G._tsem_lik_stock(P, sess, X, rho, shrink, m0, nmin)
        for s in np.unique(sess):
            r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1); Sg = SIG[per[r[0]]]
            for t in range(nmin, len(r)):
                A_, f = X[r[:t]], X[r[t]]
                m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3)
                Zm, z = (A_ - m) / sd, (f - m) / sd
                q = max(2, int(rho * t / 3)); sel = []
                for k in range(3):
                    ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
                if min(len(x) for x in sel) < 2:
                    continue
                mu = np.stack([Zm[x].sum(0) / (len(x) + m0) for x in sel])
                R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)])
                Cv = np.cov(R.T); tr = np.trace(Cv) / len(Cv)
                tgt = (1 - beta) * np.eye(len(Cv)) * tr + beta * Sg * (tr * len(Cv) / np.trace(Sg))
                Cv = (1 - shrink) * Cv + shrink * tgt
                L[r[t]] = T._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
        return L, tt
    if not hasattr(G, "_tsem_lik_stock"):
        G._tsem_lik_stock = G.tsem_lik
    G.tsem_lik = tsem_lik


def stage_consts(gm, em_lam=None, ts_lam=None, bts=None):
    """s232: the stage constants as knobs. SessionEM's lam (0.5), TSEM's base strength (0.15) and its gate boost bts (BTS)."""
    if em_lam is not None:
        def em_g(b, proto, g, bem=gm.BEM, nmin=gm.NMIN):
            F, sess = gm.L.Z[f"GE|{proto}"], gm.L.Z[f"sess|{proto}"]; out = np.empty_like(b)
            for s in dict.fromkeys(sess):
                e = gm.EMg(bem, lam=em_lam, n0=60, m0=20, shrink=0.7, nmin=nmin)
                for r in np.flatnonzero(sess == s):
                    e.g = g[r]; out[r] = e.step(F[r], b[r])
            return out
        gm.em_g = em_g
    if ts_lam is not None or bts is not None:
        tl, bt = 0.15 if ts_lam is None else ts_lam, gm.BTS if bts is None else bts

        def tsem_g(P, proto, g, bts=None):
            Lk, tt = gm.L.G.tsem_lik(P, gm.L.Z[f"sess|{proto}"], gm.L.Z[f"pc128|{proto}"], rho=0.7)
            out = P.copy(); m = ~np.isnan(Lk[:, 0])
            e = tl * tt[m] / (tt[m] + 60) * (1 + bt * (gm.M.h(tt[m], 30) * g[m]))
            q = P[m] * Lk[m] ** e[:, None]; out[m] = q / q.sum(1, keepdims=True); return out
        gm.tsem_g = tsem_g
