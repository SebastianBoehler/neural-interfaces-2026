"""OFFTASK hooks (ledger s241 step 2a): SessionEM and TSEM with each window's contribution to the template estimates
weighted by w (w = (1 - p_rest)^gamma from the rest net). Only learning changes: the current window's own likelihood and
exponent are as in the stock stages. Weighted parts: SessionEM's class means (responsibilities x w) and residual covariance
(aweights w); TSEM's seed class means (sum w z / (sum w + m0)) and residual covariance (aweights w). Seed selection
(top confident per class) is unchanged. With w = 1 both reduce to the stock code (checked by noop_check()).
install(gm, wglob, p0): wglob is indexed by global window index; the rows' indices are read from the replay store
gm.L.Z ('idx|<p0>'), which shuffle.apply permutes with every other per-window array."""
import numpy as np

_STOCK = {}


def _weights(Z, p0, wglob, n):
    idx = Z[f"idx|{p0}"].astype(int); assert len(idx) == n, (len(idx), n); return wglob[idx]


def install(gm, wglob, p0="all_loso"):
    SE, M, G, T, L = gm.SE, gm.M, gm.L.G, gm.L.G.T, gm.L
    _STOCK.setdefault("em_g", gm.em_g); _STOCK.setdefault("tsem_lik", G.tsem_lik)
    stock_em, stock_ts = _STOCK["em_g"], _STOCK["tsem_lik"]

    class WEMg(gm.EMg):
        wcur = 1.0

        def __init__(self, *a, **k):
            super().__init__(*a, **k); self.W = []

        def step(self, f, p):
            out = super().step(f, p); self.W.append(self.wcur); return out

        def _adapt(self, A, B, f, p, t):
            W = np.asarray(self.W[:t], dtype=np.float64); assert len(W) == t
            m, sd = A.mean(0), np.maximum(A.std(0), SE.SD_FLOOR)
            Z, z = (A - m) / sd, (f - m) / sd
            Q = B
            for _ in range(self.iters):
                Qw = Q * W[:, None]
                mu = (Qw / (Qw.sum(0, keepdims=True) + self.m0)).T @ Z
                C = np.cov((Z - Q @ mu).T, aweights=W)
                C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
                Ci = np.linalg.inv(C)
                Q = B * SE._softmax_lda(Z, mu, Ci) ** self.lam
                Q = Q / Q.sum(1, keepdims=True)
            e = self.lam * t / (t + self.n0) * (1 + self.bem * (M.h(t, 30) * self.g))
            q = p * SE._softmax_lda(z[None], mu, Ci)[0] ** e
            return q / q.sum()

    def em_g(b, proto, g, bem=gm.BEM, nmin=gm.NMIN):
        if proto != p0:
            return stock_em(b, proto, g, bem, nmin)
        F, sess = L.Z[f"GE|{proto}"], L.Z[f"sess|{proto}"]; out = np.empty_like(b); w = _weights(L.Z, p0, wglob, len(b))
        for s in dict.fromkeys(sess):
            e = WEMg(bem, lam=0.5, n0=60, m0=20, shrink=0.7, nmin=nmin)
            for r in np.flatnonzero(sess == s):
                e.g = g[r]; e.wcur = w[r]; out[r] = e.step(F[r], b[r])
        return out

    def tsem_lik(P, sess, X, rho=0.7, shrink=0.7, m0=20, nmin=15):
        if len(P) != len(L.Z[f"y|{p0}"]):
            return stock_ts(P, sess, X, rho, shrink, m0, nmin)
        w = _weights(L.Z, p0, wglob, len(P)); Lk = np.full_like(P, np.nan); tt = np.zeros(len(P))
        for s in np.unique(sess):
            r = np.flatnonzero(sess == s); lab = P[r].argmax(1); conf = P[r].max(1); wr = w[r]
            for t in range(nmin, len(r)):
                A_, f = X[r[:t]], X[r[t]]
                m, sd = A_.mean(0), np.maximum(A_.std(0), 1e-3)
                Zm, z = (A_ - m) / sd, (f - m) / sd
                q = max(2, int(rho * t / 3)); sel = []
                for k in range(3):
                    ik = np.flatnonzero(lab[:t] == k); sel.append(ik[np.argsort(-conf[ik])[:q]])
                if min(len(x) for x in sel) < 2:
                    continue
                mu = np.stack([(wr[x, None] * Zm[x]).sum(0) / (wr[x].sum() + m0) for x in sel])
                R = np.concatenate([Zm[x] - mu[k] for k, x in enumerate(sel)]); aw = np.concatenate([wr[x] for x in sel])
                Cv = np.cov(R.T, aweights=aw); Cv = (1 - shrink) * Cv + shrink * np.eye(len(Cv)) * np.trace(Cv) / len(Cv)
                Lk[r[t]] = T._lda(z[None], mu, np.linalg.inv(Cv))[0]; tt[r[t]] = t
        return Lk, tt

    gm.em_g = em_g; G.tsem_lik = tsem_lik


def uninstall(gm):
    if _STOCK:
        gm.em_g = _STOCK["em_g"]; gm.L.G.tsem_lik = _STOCK["tsem_lik"]


def noop_check(gm, bn="B2_pool", p0="all_loso", tol=1e-9):
    """w = 1 must reproduce the stock stages on one base (rebalanced mix -> gate -> SessionEM -> TSEM)."""
    mix = gm.L.mixw(gm.L.Z[f"{bn}|Ps|{p0}"], gm.L.Z[f"{bn}|ws|{p0}"]); b = gm.M.rebal(mix, p0, None); g = gm.gate(b, p0)
    uninstall(gm); ref = gm.tsem_g(gm.em_g(b, p0, g), p0, g)
    install(gm, np.ones(int(gm.L.Z[f"idx|{p0}"].max()) + 1), p0); got = gm.tsem_g(gm.em_g(b, p0, g), p0, g); uninstall(gm)
    d = np.abs(got - ref).max(); assert d < tol, f"w = 1 does not reproduce the stock stages ({d:.2e})"
    return d
