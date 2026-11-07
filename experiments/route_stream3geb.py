"""Routing v3 + gated early boost (GEB): self-contained streaming reference of cand_rec_ts2_tsem_route3geb's tail,
numpy only (7 Nov cleanup: route_stream3.py, route_stream3eb.py and this file's GEB layer merged, arithmetic unchanged;
the separate v1 / v2 / v3 / EB references are in commit c4d600d).

Per session (reset_state at every session start), window t: out_t, shadow_t = V3TailGEB(p).step(mix_t, core1_t, f_t, u_t)
  mix_t : member mixture before rebalancing (3,);   core1_t: core1's probabilities (3,) (its mix share = route_slot_a)
  f_t   : SessionEM features of window t;           u_t    : 256-PC view ([TSEM view, route_V_hi view]); TSEM uses u[:128]
Shadow chain (unrouted): rebalance -> SessionEM -> TSEM on mix_t -> shadow_t. The gate reads ONLY shadow outputs and
views of windows < t (robust median/MAD z clipped at +-route_clip, q70 seeds, contrast projected off
span{route_pop, route_b}, cosine with projected galleries): j = argmax, r_t = clip((s_j - tau) / ramp, 0, 1) for
t >= t_start. Routed chain: its own rebalance -> SessionEM -> TSEM on
  mix'_t = (mix_t - r a core1_t + g r a h_j(t)) / (1 - r a + g r a),  h_j = softmax(W_j z_t + c_j) (PS1 head),
z_t = u_t standardised by mean/std of windows 0..t (centred only for t < 4); mix'_t = mix_t when r_t = 0. Output = routed.
GEB (eb2): each chain's SessionEM keeps the running mean m of its INPUT margin (top1 - top2) over windows 0..t-1; gate
g(t) = clip((eb_gate_hi - m) / (eb_gate_hi - eb_gate_lo), 0, 1) (1 at t = 0 or when disabled) multiplies the early boost
h(t) = clip((2 T - t) / T, 0, 1) in that SessionEM (exponent x (1 + em_boost h g)) and in the SAME chain's TSEM
(x (1 + tsem_boost h g)). Without em_boost / tsem_boost / gate keys this is the plain v3 (or EB) reference bitwise.
Keys: route_gal, route_Q, route_names, route_tau, route_ramp, route_t_start, route_rho, route_clip, route_ps1_W (P,3,256),
route_ps1_c (P,3), route_slot_a, route_slot_g, route_v3 (= 1); rb_alpha, rb_k, rb_w; em_lam, em_n0, em_m0, em_shrink,
em_iters, em_nmin; tsem_lam, tsem_rho, tsem_shrink, tsem_m0, tsem_n0, tsem_nmin; em_boost, tsem_boost, eb_T, eb_gate_lo,
eb_gate_hi (missing: 0, 0, 30, 0, 0)."""
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


def h_early(T, t):
    return min(max((2 * T - t) / T, 0.0), 1.0)


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


class SessEMg:  # package SessionEM on given features, with the gated early boost
    def __init__(self, lo, hi, boost, T, lam, n0, m0, shrink, iters, nmin):
        self.g_lo, self.g_hi, self.boost, self.T = lo, hi, boost, T
        self.lam, self.n0, self.m0, self.shrink, self.iters, self.nmin = lam, n0, m0, shrink, iters, nmin
        self.reset_state()

    def reset_state(self):
        self.A, self.B = [], []
        self.msum, self.gates = 0.0, []

    def _gate(self, t):
        if self.g_hi <= self.g_lo or t == 0:
            return 1.0
        return min(max((self.g_hi - self.msum / t) / (self.g_hi - self.g_lo), 0.0), 1.0)

    def step(self, f, p):
        t, out = len(self.A), p
        self.gates.append(self._gate(t))
        if t >= self.nmin:
            try:
                out = self._adapt(np.asarray(self.A), np.asarray(self.B), f, p, t)
            except np.linalg.LinAlgError:
                pass
            if not np.isfinite(out).all():
                out = p
        self.A.append(f); self.B.append(p)
        s = np.sort(p)
        self.msum += s[-1] - s[-2]
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
        e *= 1 + self.boost * (h_early(self.T, t) * self.gates[t])
        q = p * lik(z[None], mu, Ci)[0] ** e
        return q / q.sum()


class TSEMsg:  # package TSEM on a given view, boosted with the gate of the same chain's SessEMg (gate_src)
    gate_src = None

    def __init__(self, boost, T, lam, rho, shrink, m0, n0, nmin):
        self.boost, self.T = boost, T
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
        g = self.gate_src.gates[t] if self.gate_src is not None and t < len(self.gate_src.gates) else 1.0
        e *= 1 + self.boost * (h_early(self.T, t) * g)
        qq = q * lik(z[None], mu, np.linalg.inv(C))[0] ** e
        return qq / qq.sum()


class ChainG:
    def __init__(self, p):
        g = lambda k: float(p[k])
        T, lo, hi = float(p.get("eb_T", 30.0)), float(p.get("eb_gate_lo", 0.0)), float(p.get("eb_gate_hi", 0.0))
        self.rb = Rebal(g("rb_alpha"), g("rb_k"), int(p["rb_w"]))
        self.em = SessEMg(lo, hi, float(p.get("em_boost", 0.0)), T, g("em_lam"), g("em_n0"), g("em_m0"),
                          g("em_shrink"), int(p["em_iters"]), int(p["em_nmin"]))
        self.ts = TSEMsg(float(p.get("tsem_boost", 0.0)), T, g("tsem_lam"), g("tsem_rho"), g("tsem_shrink"),
                         g("tsem_m0"), g("tsem_n0"), int(p["tsem_nmin"]))
        self.ts.gate_src = self.em

    def reset_state(self):
        self.rb.reset_state(); self.em.reset_state(); self.ts.reset_state()

    def step(self, mix, f, v):
        return self.ts.step(v, self.em.step(f, self.rb.step(mix)))


class V3TailGEB:
    def __init__(self, p):
        f = lambda k: np.asarray(p[k], dtype=np.float64)
        assert int(p["route_v3"]) == 1
        self.Qb, self.W, self.c = f("route_Q"), f("route_ps1_W"), f("route_ps1_c")
        self.Gn = gal_norm(f("route_gal"), self.Qb)
        self.tau, self.ramp, self.rho, self.clip = (float(p[f"route_{k}"]) for k in ("tau", "ramp", "rho", "clip"))
        self.a, self.g, self.t_start = float(p["route_slot_a"]), float(p["route_slot_g"]), int(p["route_t_start"])
        self.shadow, self.routed = ChainG(p), ChainG(p)
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
