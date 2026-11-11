"""Gated EB in the early/ (retune inputs) framework, for routing v3 and package parity: gate from the SessionEM input
(rebalanced mix) of the same chain; margin = top1 - top2, causal session mean over windows < t."""
import paths  # noqa: F401
import numpy as np
import lib as L, mech as M  # early/lib.py, early/mech.py
import session_em as SE

LO, HI, BEM, NMIN, BTS = 0.20, 0.35, 2.0, 20, 4.0


def gate(b, proto, lo=LO, hi=HI):
    sess = L.Z[f"sess|{proto}"]; s = np.sort(b, 1); v = s[:, -1] - s[:, -2]; g = np.ones(len(b))
    for x in dict.fromkeys(sess):
        r = np.flatnonzero(sess == x); cs = np.concatenate([[0.0], np.cumsum(v[r])]); t = np.arange(len(r))
        with np.errstate(divide="ignore", invalid="ignore"):
            gg = np.clip((hi - cs[t] / np.maximum(t, 1)) / (hi - lo), 0.0, 1.0)
        g[r] = np.where(t == 0, 1.0, gg)
    return g


class EMg(M.EMb):
    g = 1.0

    def _adapt(self, A, B, f, p, t):
        m, sd = A.mean(0), np.maximum(A.std(0), SE.SD_FLOOR)
        Z, z = (A - m) / sd, (f - m) / sd
        Q = B
        for _ in range(self.iters):
            mu = (Q / (Q.sum(0, keepdims=True) + self.m0)).T @ Z
            C = np.cov((Z - Q @ mu).T)
            C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
            Ci = np.linalg.inv(C)
            Q = B * SE._softmax_lda(Z, mu, Ci) ** self.lam
            Q = Q / Q.sum(1, keepdims=True)
        e = self.lam * t / (t + self.n0) * (1 + self.bem * (M.h(t, 30) * self.g))
        q = p * SE._softmax_lda(z[None], mu, Ci)[0] ** e
        return q / q.sum()


def em_g(b, proto, g, bem=BEM, nmin=NMIN):
    F, sess = L.Z[f"GE|{proto}"], L.Z[f"sess|{proto}"]; out = np.empty_like(b)
    for s in dict.fromkeys(sess):
        e = EMg(bem, lam=0.5, n0=60, m0=20, shrink=0.7, nmin=nmin)
        for r in np.flatnonzero(sess == s):
            e.g = g[r]; out[r] = e.step(F[r], b[r])
    return out


def tsem_g(P, proto, g, bts=BTS):
    Lk, tt = L.G.tsem_lik(P, L.Z[f"sess|{proto}"], L.Z[f"pc128|{proto}"], rho=0.7)
    out = P.copy(); m = ~np.isnan(Lk[:, 0])
    e = 0.15 * tt[m] / (tt[m] + 60) * (1 + bts * (M.h(tt[m], 30) * g[m]))
    q = P[m] * Lk[m] ** e[:, None]; out[m] = q / q.sum(1, keepdims=True); return out


def pipe_g(mix, proto):
    b = M.rebal(mix, proto, None); g = gate(b, proto); return tsem_g(em_g(b, proto, g), proto, g)
