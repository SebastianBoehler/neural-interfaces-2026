"""Class-direction routing stage (route team, 6 Nov): self-contained streaming reference, numpy only.

Runs AFTER the TSEM stage, window by window, per session (reset_state at every session start):
    out_t = RouteStream.step(u_t, q_t)
u_t: the TSEM view of window t, (z_t - tsem_m) @ V256 (core1 scaled causal features on the final PC256 basis), (256,);
q_t: the TSEM output posterior of window t, (3,). Returns the routed posterior (3,); q_t itself while not routed.

Window t (0-based) uses u_0..u_t and q_0..q_{t-1} only (causal). From t >= route_t_start: probe class contrast from
windows 0..t-1 (per-session standardised u, q70 seeds of the stored TSEM outputs: argmax label, the max(2, int(rho t/3))
most confident per class), minus the debias vector; score_p = cos(c - pop, gallery_p - pop) for each known person;
p* = argmax; w = wmax * clip((score_p* - tau) / ramp, 0, 1); if w > 0: out = normalise(exp(log q + w log h_p*(t))),
h_p = softmax(W_p z + c_p) with z = u_t standardised over windows 0..t (centred only for t < 4).
Keys (npz): route_gal (P,3,D), route_pop (3,D), route_b (3,D), route_head_W (P,3,D), route_head_c (P,3),
route_names (P,), route_tau, route_ramp, route_wmax, route_t_start, route_rho."""
import numpy as np


def _blend(q, h, w):
    L = np.log(np.clip(q, 1e-9, 1)) + w * np.log(np.clip(h, 1e-9, 1))
    L -= L.max(-1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(-1, keepdims=True)


def _softmax_rows(D):  # sklearn.utils.extmath.softmax (LogisticRegression.predict_proba, multinomial)
    D = D - np.max(D, axis=1).reshape((-1, 1))
    np.exp(D, D)
    return D / np.sum(D, axis=1).reshape((-1, 1))


class RouteStream:
    def __init__(self, p):
        f = lambda k: np.asarray(p[k], dtype=np.float64)
        self.G, self.pop, self.b = f("route_gal"), f("route_pop"), f("route_b")
        self.W, self.c = f("route_head_W"), f("route_head_c")
        self.tau, self.ramp, self.wmax, self.rho = (float(p[f"route_{k}"]) for k in ("tau", "ramp", "wmax", "rho"))
        self.t_start = int(p["route_t_start"])
        self.Gr = [(g - self.pop).ravel() for g in self.G]
        self.reset_state()

    def reset_state(self):
        self.U, self.Q = [], []
        self.last = None  # (person index, score, weight) of the last window, for logging

    def scores(self, t):
        """Residual-cosine scores (P,) from windows 0..t-1, or None if a class has no seed."""
        X, P = np.asarray(self.U[:t]), np.asarray(self.Q[:t])
        lab, conf = P.argmax(1), P.max(1); Wt = np.zeros((t, 3)); q = max(2, int(self.rho * t / 3))
        for k in range(3):
            ik = np.flatnonzero(lab == k); Wt[ik[np.argsort(-conf[ik])[:q]], k] = 1
        m = X.mean(0); Z = (X - m) / np.maximum(X.std(0), 1e-6)
        mu = []
        for k in range(3):
            a = Wt[:, k]
            if a.sum() < 1e-9:
                return None
            mu.append((a[:, None] * Z).sum(0) / a.sum())
        mu = np.stack(mu); c = mu - mu.mean(0) - self.b
        a = (c - self.pop).ravel(); na = np.linalg.norm(a)
        return np.array([float(a @ g / (na * np.linalg.norm(g) + 1e-12)) for g in self.Gr])

    def step(self, u, q):
        u, q = np.asarray(u, dtype=np.float64), np.asarray(q, dtype=np.float64)
        t, out = len(self.U), q
        self.last = None
        if t >= self.t_start:
            try:
                s = self.scores(t)
                if s is not None and np.isfinite(s).all():
                    j = int(np.argmax(s)); w = self.wmax * np.clip((s[j] - self.tau) / self.ramp, 0, 1)
                    self.last = (j, s[j], w)
                    if w > 0:
                        A = np.asarray(self.U + [u]); m = A.mean(0)
                        z = (u - m) / np.maximum(A.std(0), 1e-3) if t >= 4 else u - m
                        h = _softmax_rows(z[None] @ self.W[j].T + self.c[j])[0]
                        o = _blend(q, h, w)
                        if np.isfinite(o).all():
                            out = o
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                out = q
        self.U.append(u); self.Q.append(q)
        return out


def run_batch(U, Q, sess, p):
    """Batch equivalent: per session (ids in sess), stream windows in order with reset_state."""
    rs = RouteStream(p); out = np.empty_like(Q)
    for s in dict.fromkeys(sess):
        rs.reset_state()
        for i in np.flatnonzero(sess == s):
            out[i] = rs.step(U[i], Q[i])
    return out
