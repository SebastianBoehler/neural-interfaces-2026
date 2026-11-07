"""Routing v3 + early boost (EB, 7 Nov): streaming reference = route_stream3.V3Tail with the early-window boost in BOTH
tails (shadow and routed), as early/combo.py injects it and as the package's Route3.bind inherits it (deep copies of the
boosted SessionEM / TSEM). Boost (early/mech.py): each stage's output exponent e is multiplied by
1 + boost * clip((2 T - t) / T, 0, 1) (SessionEM: em_boost; TSEM: tsem_boost; T = eb_T); em_nmin is read from the keys.
Keys: route_stream3's plus em_boost, tsem_boost, eb_T (missing: 0, 0, 30, i.e. identical to V3Tail)."""
import numpy as np
import route_stream3 as R3


def gain(boost, T, t):
    return 1 + boost * min(max((2 * T - t) / T, 0.0), 1.0)


class SessEMb(R3.SessEM):
    def __init__(self, boost, T, *a):
        self.boost, self.T = boost, T
        super().__init__(*a)

    def _adapt(self, A, B, f, p, t):  # = R3.SessEM._adapt with the boosted output exponent
        m, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
        Z, z = (A - m) / sd, (f - m) / sd
        Q = B
        for _ in range(self.iters):
            mu = (Q / (Q.sum(0, keepdims=True) + self.m0)).T @ Z
            C = np.cov((Z - Q @ mu).T)
            C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
            Ci = np.linalg.inv(C)
            Q = B * R3.lik(Z, mu, Ci) ** self.lam
            Q = Q / Q.sum(1, keepdims=True)
        e = self.lam * t / (t + self.n0) if self.n0 else self.lam
        e *= gain(self.boost, self.T, t)
        q = p * R3.lik(z[None], mu, Ci)[0] ** e
        return q / q.sum()


class TSEMsb(R3.TSEMs):
    def __init__(self, boost, T, *a):
        self.boost, self.T = boost, T
        super().__init__(*a)

    def _adapt(self, A, conf, lab, u, q, t):  # = R3.TSEMs._adapt with the boosted output exponent
        m, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
        Z, z = (A - m) / sd, (u - m) / sd
        k = max(2, int(self.rho * t / 3))
        sel = [i[np.argsort(-conf[i])[:k]] for i in (np.flatnonzero(lab == c) for c in range(3))]
        if min(len(i) for i in sel) < 2:
            return q
        mu = np.stack([Z[i].sum(0) / (len(i) + self.m0) for i in sel])
        C = np.cov(np.concatenate([Z[i] - mu[c] for c, i in enumerate(sel)]).T)
        C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
        e = self.lam * t / (t + self.n0) if self.n0 else self.lam
        e *= gain(self.boost, self.T, t)
        qq = q * R3.lik(z[None], mu, np.linalg.inv(C))[0] ** e
        return qq / qq.sum()


class ChainEB(R3.Chain):
    def __init__(self, p):
        g = lambda k: float(p[k])
        T = float(p.get("eb_T", 30.0))
        self.rb = R3.Rebal(g("rb_alpha"), g("rb_k"), int(p["rb_w"]))
        self.em = SessEMb(float(p.get("em_boost", 0.0)), T, g("em_lam"), g("em_n0"), g("em_m0"), g("em_shrink"),
                          int(p["em_iters"]), int(p["em_nmin"]))
        self.ts = TSEMsb(float(p.get("tsem_boost", 0.0)), T, g("tsem_lam"), g("tsem_rho"), g("tsem_shrink"),
                         g("tsem_m0"), g("tsem_n0"), int(p["tsem_nmin"]))


class V3TailEB(R3.V3Tail):
    def __init__(self, p):
        super().__init__(p)
        self.shadow, self.routed = ChainEB(p), ChainEB(p)
        self.reset_state()
