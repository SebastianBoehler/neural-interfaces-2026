"""Position-dependent chain (families A/B/C of families.txt). run(base, proto, cfg) -> TSEM output (n,3).
cfg keys (all optional; missing = shipped): A=(target, rho, T); B=(bem, em_nmin, bts); C=(pi0 (3,) or None, ke, ae)."""
import numpy as np
import lib as L
import session_em as SE

SLOT, SBN = [0], [5, 6, 7, 8, 9]


SHIFT = 0  # control only: boost profile moved to windows SHIFT..SHIFT+60 (0 = the mechanism)


def h(t, T):
    t = np.asarray(t, float) - SHIFT
    return np.where(t >= 0, np.clip((2 * T - t) / T, 0, 1), 0.0)


def weights(base, proto, A):
    ws = L.Z[f"{base}|ws|{proto}"]; t = L.pos(proto)
    if A is None:
        return ws
    target, rho, T = A; W = np.tile(ws, (len(t), 1)); ix = SLOT + (SBN if target == "slot+sbn" else [])
    W[:, ix] *= (1 + (rho - 1) * h(t, T))[:, None]
    return W


class EMb(SE.SessionEM):
    def __init__(self, bem, **kw):
        super().__init__(**kw); self.bem = bem

    def _adapt(self, A, B, f, p, t):  # = SE.SessionEM._adapt with the output exponent boosted early
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
        e = self.lam * t / (t + self.n0) * (1 + self.bem * h(t, 30))
        q = p * SE._softmax_lda(z[None], mu, Ci)[0] ** e
        return q / q.sum()


def em(P, proto, bem=0.0, nmin=10):
    F, sess = L.Z[f"GE|{proto}"], L.Z[f"sess|{proto}"]; out = np.empty_like(P)
    for s in dict.fromkeys(sess):
        e = EMb(bem, lam=0.5, n0=60, m0=20, shrink=0.7, nmin=nmin)
        for r in np.flatnonzero(sess == s):
            out[r] = e.step(F[r], P[r])
    return out


def tsem(P, proto, bts=0.0):
    Lk, tt = L.G.tsem_lik(P, L.Z[f"sess|{proto}"], L.Z[f"pc128|{proto}"], rho=0.7)
    out = P.copy(); m = ~np.isnan(Lk[:, 0])
    e = 0.15 * tt[m] / (tt[m] + 60) * (1 + bts * h(tt[m], 30))
    q = P[m] * Lk[m] ** e[:, None]; out[m] = q / q.sum(1, keepdims=True)
    return out


def global_pi0(base, exclude):
    """session-equal mean of the mean mix posterior over t < 60, all_loso sessions of people != exclude."""
    proto = "all_loso"; mix = L.mixw(L.Z[f"{base}|Ps|{proto}"], L.Z[f"{base}|ws|{proto}"])
    sess, per, t = L.Z[f"sess|{proto}"], L.Z[f"person|{proto}"], L.pos(proto)
    v = [mix[(sess == s) & (t < 60)].mean(0) for s in np.unique(sess) if per[sess == s][0] != exclude]
    return np.mean(v, 0)


def rebal(mix, proto, C):
    sess = L.Z[f"sess|{proto}"]
    if C is None:
        return L.rebal(mix, sess)
    pi0, ke, ae = C; u = np.full(3, 1 / 3); pi0 = u if pi0 is None else np.asarray(pi0)
    out = np.empty_like(mix)
    for s in np.unique(sess):  # = L.rebal with per-position prior / k / alpha
        r = np.flatnonzero(sess == s); X = mix[r]
        cs = np.vstack([np.zeros((1, 3)), np.cumsum(X, 0)])
        n = np.arange(len(X)); lo = np.maximum(0, n - 30); hh = h(n, 30)[:, None]
        k = (1 - hh) * 5 + hh * ke; a = (1 - hh) * 1.15 + hh * ae; pr = (1 - hh) * u + hh * pi0
        m = (k * pr + cs[n] - cs[lo]) / (k + (n - lo)[:, None])
        q = X / m ** a
        out[r] = q / q.sum(1, keepdims=True)
    return out


def run(base, proto, cfg, stages=False):
    A, B, C = cfg.get("A"), cfg.get("B"), cfg.get("C")
    mix = L.mixw(L.Z[f"{base}|Ps|{proto}"], weights(base, proto, A))
    b = rebal(mix, proto, C)
    e = em(b, proto, *(B[:2] if B else (0.0, 10)))
    t = tsem(e, proto, B[2] if B else 0.0)
    return (mix, b, e, t) if stages else t
