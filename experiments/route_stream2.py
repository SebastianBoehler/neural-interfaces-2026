"""Returning-person routing stage v2 (route2, 7 Nov): self-contained streaming reference, numpy only.

Runs AFTER the TSEM stage, window by window, per session (reset_state at every session start):
    out_t = RouteStream.step(u_t, q_t)
u_t: the 256-PC view of window t ([TSEM view, (z_t - tsem_m) @ route_V_hi]); q_t: TSEM output posterior (3,).
Window t uses u_0..u_t and q_0..q_{t-1} only (causal). Changes vs v1 (review 7 Nov: P04 attractor, artifact windows):
  * robust standardisation: per-feature median and 1.4826 * MAD over the session's windows, z clipped at +-route_clip
    (probe contrast over windows 0..t-1, head input over windows 0..t);
  * shape score: score_p = cos(P c, P G_p), P = projector onto the orthogonal complement of span{route_pop, route_b}
    (route_Q, an orthonormal (3D, 2) basis). v1 compared c - b - pop with G_p - pop: a session with weak or no class
    contrast then lands on -(b + pop), which is closest to the weakest gallery. Here a zero contrast scores 0, and
    scaling the contrast does not change the score.
From t >= route_t_start: p* = argmax_p score_p; w = wmax * clip((score_p* - tau) / ramp, 0, 1); if w > 0:
out = normalise(exp(log q + w log h_p*(t))), h_p = softmax(W_p z + c_p). Otherwise out = q_t.
Keys (npz): route_gal (P,3,D), route_pop (3,D), route_b (3,D), route_Q (3D,2), route_head_W (P,3,D), route_head_c (P,3),
route_names (P,), route_tau, route_ramp, route_wmax, route_t_start, route_rho, route_clip, route_v2 (= 1)."""
import numpy as np


def rz(A, x, clip, floor):
    """x standardised by the per-feature median / (1.4826 MAD) of the rows of A, clipped at +-clip."""
    med = np.median(A, axis=0)
    s = 1.4826 * np.median(np.abs(A - med), axis=0)
    return np.clip((x - med) / np.maximum(s, floor), -clip, clip)


def seeds(P, rho):
    """(t, 3) 0/1 seed mask: per argmax class, the max(2, int(rho t / 3)) most confident rows."""
    t = len(P); lab, conf = P.argmax(1), P.max(1); Wt = np.zeros((t, 3)); q = max(2, int(rho * t / 3))
    for k in range(3):
        ik = np.flatnonzero(lab == k)
        Wt[ik[np.argsort(-conf[ik])[:q]], k] = 1
    return Wt


def probe(X, P, rho, clip):
    """Class contrast (3, D) of the seeded rows of X (robust-standardised over X), or None if a class has no seed."""
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


def blend(q, h, w):
    L = np.log(np.clip(q, 1e-9, 1)) + w * np.log(np.clip(h, 1e-9, 1))
    L -= L.max(-1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(-1, keepdims=True)


def softmax_rows(D):  # sklearn.utils.extmath.softmax (LogisticRegression.predict_proba, multinomial)
    D = D - np.max(D, axis=1).reshape((-1, 1))
    np.exp(D, D)
    return D / np.sum(D, axis=1).reshape((-1, 1))


class RouteStream:
    def __init__(self, p):
        f = lambda k: np.asarray(p[k], dtype=np.float64)
        assert int(p["route_v2"]) == 1
        self.G, self.W, self.c, self.Qb = f("route_gal"), f("route_head_W"), f("route_head_c"), f("route_Q")
        self.tau, self.ramp, self.wmax, self.rho, self.clip = (float(p[f"route_{k}"]) for k in ("tau", "ramp", "wmax", "rho", "clip"))
        self.t_start = int(p["route_t_start"])
        self.Gn = gal_norm(self.G, self.Qb)
        self.reset_state()

    def reset_state(self):
        self.U, self.Q = [], []
        self.last = None  # (person index, score, weight) of the last window, for logging

    def scores(self, t):
        c = probe(np.asarray(self.U[:t]), np.asarray(self.Q[:t]), self.rho, self.clip)
        return None if c is None else shape_scores(c, self.Gn, self.Qb)

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
                        z = rz(np.asarray(self.U + [u]), u, self.clip, 1e-3)
                        h = softmax_rows(z[None] @ self.W[j].T + self.c[j])[0]
                        o = blend(q, h, w)
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
