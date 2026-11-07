"""Routing v3 (route2, 7 Nov): self-contained streaming reference of the B2 tail with a shadow chain, numpy only.

Per session (reset_state at every session start), window t: out_t, shadow_t = V3Tail.step(mix_t, core1_t, f_t, u_t)
  mix_t   : B2 member mixture before rebalancing (3,);  core1_t: core1's probabilities (3,) (its mix share = slot_a)
  f_t     : SessionEM features of window t;            u_t    : 256-PC view ([TSEM view, route_V_hi view]); TSEM uses u[:128]
Shadow chain (unrouted): rebalance -> SessionEM -> TSEM on mix_t -> shadow_t. The gate reads ONLY shadow outputs and
views of windows < t (v2: robust median/MAD z clipped at +-route_clip, q70 seeds, contrast projected off
span{route_pop, route_b}, cosine with projected galleries): j = argmax, r_t = clip((s_j - tau) / ramp, 0, 1) for
t >= t_start. Routed chain: its own rebalance -> SessionEM -> TSEM on
  mix'_t = (mix_t - r a core1_t + g r a h_j(t)) / (1 - r a + g r a),  h_j = softmax(W_j z_t + c_j) (PS1 head),
z_t = u_t standardised by mean/std of windows 0..t (centred only for t < 4); mix'_t = mix_t when r_t = 0, so the routed
chain equals the shadow chain until the first routed window. Output = routed chain.
Keys: v2 gate keys (route_gal, route_Q, route_names, route_tau, route_ramp, route_t_start, route_rho, route_clip),
route_ps1_W (P,3,256), route_ps1_c (P,3), route_slot_a, route_slot_g, route_v3 (= 1), plus rb_alpha, rb_k, rb_w,
em_lam, em_n0, em_m0, em_shrink, em_iters, em_nmin, tsem_lam, tsem_rho, tsem_shrink, tsem_m0, tsem_n0, tsem_nmin."""
import numpy as np


def rz(A, x, clip, floor):
    med = np.median(A, axis=0)
    s = 1.4826 * np.median(np.abs(A - med), axis=0)
    return np.clip((x - med) / np.maximum(s, floor), -clip, clip)


def seeds(P, rho):
    t = len(P); lab, conf = P.argmax(1), P.max(1); Wt = np.zeros((t, 3)); q = max(2, int(rho * t / 3))
    for k in range(3):
        ik = np.flatnonzero(lab == k)
        Wt[ik[np.argsort(-conf[ik])[:q]], k] = 1
    return Wt


def probe(X, P, rho, clip):
    Wt = seeds(P, rho); n = Wt.sum(0)
    if (n < 1e-9).any():
        return None
    Z = rz(X, X, clip, 1e-6); mu = (Wt.T @ Z) / n[:, None]
    return mu - mu.mean(0)


def perp(v, Qb):
    v = v.ravel()
    return v - Qb @ (Qb.T @ v)


def shape_scores(c, Gn, Qb):
    a = perp(c, Qb); na = np.linalg.norm(a)
    return Gn @ a / na if na > 1e-9 else np.zeros(len(Gn))


def gal_norm(G, Qb):
    A = np.stack([perp(g, Qb) for g in G])
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def softmax_rows(D):
    D = D - np.max(D, axis=1).reshape((-1, 1))
    np.exp(D, D)
    return D / np.sum(D, axis=1).reshape((-1, 1))


def lik(Z, mu, Ci):
    W = Ci @ mu.T
    s = Z @ W - 0.5 * (mu * W.T).sum(1)
    s = s - s.max(-1, keepdims=True)
    g = np.exp(s)
    return g / g.sum(-1, keepdims=True)


class Rebal:  # Ensemble.predict rebalancing (W > 0: last W rows)
    def __init__(self, alpha, k, W):
        self.alpha, self.k, self.W = alpha, k, W
        self.reset_state()

    def reset_state(self):
        self.seen, self.n, self.hist = np.zeros(3), 0, []

    def step(self, p):
        if self.W:
            h = self.hist[-self.W:]
            m = (self.k / len(p) + np.asarray(h).sum(0)) / (self.k + len(h))
        else:
            m = (self.k / len(p) + self.seen) / (self.k + self.n)
        q = p / m ** self.alpha
        self.seen, self.n = self.seen + p, self.n + 1
        if self.W:
            self.hist = (self.hist + [p])[-self.W:]
        return q / q.sum()


class SessEM:  # SessionEM on given features (package / experiments/session_em.py semantics)
    def __init__(self, lam, n0, m0, shrink, iters, nmin):
        self.lam, self.n0, self.m0, self.shrink, self.iters, self.nmin = lam, n0, m0, shrink, iters, nmin
        self.reset_state()

    def reset_state(self):
        self.A, self.B = [], []

    def step(self, f, p):
        t, out = len(self.A), p
        if t >= self.nmin:
            try:
                out = self._adapt(np.asarray(self.A), np.asarray(self.B), f, p, t)
            except np.linalg.LinAlgError:
                pass
            if not np.isfinite(out).all():
                out = p
        self.A.append(f); self.B.append(p)
        return out

    def _adapt(self, A, B, f, p, t):
        m, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
        Z, z = (A - m) / sd, (f - m) / sd
        Q = B
        for _ in range(self.iters):
            mu = (Q / (Q.sum(0, keepdims=True) + self.m0)).T @ Z
            C = np.cov((Z - Q @ mu).T)
            C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
            Ci = np.linalg.inv(C)
            Q = B * lik(Z, mu, Ci) ** self.lam
            Q = Q / Q.sum(1, keepdims=True)
        e = self.lam * t / (t + self.n0) if self.n0 else self.lam
        q = p * lik(z[None], mu, Ci)[0] ** e
        return q / q.sum()


class TSEMs:  # package TSEM semantics on a given view
    def __init__(self, lam, rho, shrink, m0, n0, nmin):
        self.lam, self.rho, self.shrink, self.m0, self.n0, self.nmin = lam, rho, shrink, m0, n0, nmin
        self.reset_state()

    def reset_state(self):
        self.U, self.conf, self.lab = [], [], []

    def step(self, u, q):
        t, out = len(self.U), q
        if self.lam > 0 and t >= self.nmin:
            try:
                out = self._adapt(np.asarray(self.U), np.asarray(self.conf), np.asarray(self.lab), u, q, t)
            except np.linalg.LinAlgError:
                out = q
            if not np.isfinite(out).all():
                out = q
        self.U.append(u); self.conf.append(q.max()); self.lab.append(q.argmax())
        return out

    def _adapt(self, A, conf, lab, u, q, t):
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
        qq = q * lik(z[None], mu, np.linalg.inv(C))[0] ** e
        return qq / qq.sum()


class Chain:
    def __init__(self, p):
        g = lambda k: float(p[k])
        self.rb = Rebal(g("rb_alpha"), g("rb_k"), int(p["rb_w"]))
        self.em = SessEM(g("em_lam"), g("em_n0"), g("em_m0"), g("em_shrink"), int(p["em_iters"]), int(p["em_nmin"]))
        self.ts = TSEMs(g("tsem_lam"), g("tsem_rho"), g("tsem_shrink"), g("tsem_m0"), g("tsem_n0"), int(p["tsem_nmin"]))

    def reset_state(self):
        self.rb.reset_state(); self.em.reset_state(); self.ts.reset_state()

    def step(self, mix, f, v):
        return self.ts.step(v, self.em.step(f, self.rb.step(mix)))


class V3Tail:
    def __init__(self, p):
        f = lambda k: np.asarray(p[k], dtype=np.float64)
        assert int(p["route_v3"]) == 1
        self.Qb, self.W, self.c = f("route_Q"), f("route_ps1_W"), f("route_ps1_c")
        self.Gn = gal_norm(f("route_gal"), self.Qb)
        self.tau, self.ramp, self.rho, self.clip = (float(p[f"route_{k}"]) for k in ("tau", "ramp", "rho", "clip"))
        self.a, self.g, self.t_start = float(p["route_slot_a"]), float(p["route_slot_g"]), int(p["route_t_start"])
        self.shadow, self.routed = Chain(p), Chain(p)
        self.reset_state()

    def reset_state(self):
        self.shadow.reset_state(); self.routed.reset_state()
        self.U, self.Qs, self.last = [], [], None  # last: (person index, score, r) of the last window

    def gate(self, t):
        c = probe(np.asarray(self.U[:t]), np.asarray(self.Qs[:t]), self.rho, self.clip)
        return None if c is None else shape_scores(c, self.Gn, self.Qb)

    def step(self, mix, core1, f, u):
        mix, core1, u = (np.asarray(x, dtype=np.float64) for x in (mix, core1, u))
        t = len(self.U); qs = self.shadow.step(mix, f, u[:128]); m2 = mix; self.last = None
        if t >= self.t_start:
            try:
                s = self.gate(t)
                if s is not None and np.isfinite(s).all():
                    j = int(np.argmax(s)); r = float(np.clip((s[j] - self.tau) / self.ramp, 0, 1)); self.last = (j, s[j], r)
                    if r > 0:
                        A = np.asarray(self.U + [u]); m = A.mean(0)
                        z = (u - m) / np.maximum(A.std(0), 1e-3) if t >= 4 else u - m
                        h = softmax_rows(z[None] @ self.W[j].T + self.c[j])[0]
                        o = (mix - r * self.a * core1 + self.g * r * self.a * h) / (1 - r * self.a + self.g * r * self.a)
                        if np.isfinite(o).all():
                            m2 = o
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                m2 = mix
        out = self.routed.step(m2, f, u[:128])
        self.U.append(u); self.Qs.append(qs)
        return out, qs
