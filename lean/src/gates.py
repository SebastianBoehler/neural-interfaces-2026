"""Returning-person gates. Each returns (public person index, score, ramp in [0, 1]) or None for no decision."""
import numpy as np


def ramp(score, tau, width):
    return float(np.clip((score - tau) / width, 0, 1))


class ContrastGate:
    """Class-direction gate, from window ``t_start``: in the 256-PC view (robustly standardised over the session so
    far), the contrast between the class means of the most confident earlier windows, taken off the population
    directions, is compared by cosine with every public person's gallery contrast."""

    def __init__(self, z):
        f = lambda k: np.asarray(z[k], dtype=np.float64)
        self.Qb = f("route_Q")
        self.tau, self.width, self.rho, self.clip = (float(z[f"route_{k}"]) for k in ("tau", "ramp", "rho", "clip"))
        self.t_start = int(z["route_t_start"])
        A = np.stack([self._perp(g) for g in f("route_gal")])
        self.gallery = A / np.linalg.norm(A, axis=1, keepdims=True)

    def _perp(self, v):
        v = v.ravel()
        return v - self.Qb @ (self.Qb.T @ v)

    def step(self, U, Q):
        """U: views of the earlier windows; Q: their unrouted posteriors."""
        t = len(U)
        if t < self.t_start:
            return None
        X, P = np.asarray(U), np.asarray(Q)
        lab, conf = P.argmax(1), P.max(1)
        Wt, q = np.zeros((t, 3)), max(2, int(self.rho * t / 3))
        for k in range(3):
            ik = np.flatnonzero(lab == k)
            Wt[ik[np.argsort(-conf[ik])[:q]], k] = 1
        n = Wt.sum(0)
        if (n < 1e-9).any():
            return None
        med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - med), axis=0)
        Z = np.clip((X - med) / np.maximum(mad, 1e-6), -self.clip, self.clip)
        mu = (Wt.T @ Z) / n[:, None]
        a = self._perp(mu - mu.mean(0))
        na = np.linalg.norm(a)
        s = self.gallery @ a / na if na > 1e-9 else np.zeros(len(self.gallery))
        if not np.isfinite(s).all():
            return None
        j = int(np.argmax(s))
        return j, s[j], ramp(s[j], self.tau, self.width)


class IdentityGate:
    """Label-free identity gate, from window ``t0``: the session's mean log-spectral shape, projected, is scored
    against every public person's gallery mean under a two-covariance model (between people B, within person W,
    within session S); the score is the log likelihood ratio same person / different person per dimension."""

    def __init__(self, z):
        f = lambda k: np.asarray(z[k], dtype=np.float64)
        self.mu, self.sd, self.V, self.B, self.W, self.S = (f(k) for k in ("bio_mu", "bio_sd", "bio_V", "bio_B", "bio_W", "bio_S"))
        self.G, self.m = f("bio_gal"), np.asarray(z["bio_m"]).astype(int)  # gallery means and their session counts
        self.tau, self.width, self.t0 = float(z["bio_tau"]), float(z["bio_ramp"]), int(z["bio_t0"])
        self.win, self.d = np.hanning(120), self.V.shape[1]
        self.A = {m: self.B + self.W / m for m in np.unique(self.m)}
        self.gAg = np.array([g @ np.linalg.solve(self.A[m], g) for g, m in zip(self.G, self.m)])
        self.ldA = {m: np.linalg.slogdet(A)[1] for m, A in self.A.items()}
        self.reset_state()

    def reset_state(self):
        self.sum, self.n = 0.0, 0

    def feat(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = x - x.mean(0, keepdims=True)
        x = x - x.mean(1, keepdims=True)
        seg = np.stack([x[:, s:s + 120] for s in range(0, 361, 60)], 1) * self.win
        L = np.log((np.abs(np.fft.rfft(seg, axis=-1)) ** 2).mean(1)[:, 2:41] + 1e-12)
        return (L - L.mean(1, keepdims=True)).ravel()

    def scores(self):
        n, d = self.n, self.d
        x = ((self.sum / n - self.mu) / self.sd) @ self.V
        C = self.B + self.W + self.S / n
        xCx, ldC = x @ np.linalg.solve(C, x), np.linalg.slogdet(C)[1]
        out = np.empty(len(self.G))
        for m, A in self.A.items():
            same = np.block([[A, self.B], [self.B, C]])
            ld = np.linalg.slogdet(same)[1]
            for i in np.flatnonzero(self.m == m):
                v = np.concatenate([self.G[i], x])
                out[i] = (-0.5 * (v @ np.linalg.solve(same, v) + ld) + 0.5 * (self.gAg[i] + xCx + self.ldA[m] + ldC)) / d
        return out

    def step(self, x):
        self.sum = self.sum + self.feat(x)
        self.n += 1
        if self.n - 1 < self.t0:
            return None
        s = self.scores()
        if not np.isfinite(s).all():
            return None
        j = int(np.argmax(s))
        return j, float(s[j]), ramp(s[j], self.tau, self.width)
