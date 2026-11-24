"""Lane Q: vectorised IdentityGate scoring (spectral identity gate of the packages) + cohort score normalisation. numpy only."""
import numpy as np


class Gate:
    def __init__(self, z):
        f = lambda k: np.asarray(z[k], np.float64)
        self.mu, self.sd, self.V = f("bio_mu"), f("bio_sd"), f("bio_V")
        self.B, self.W, self.S = f("bio_B"), f("bio_W"), f("bio_S")
        self.G0, self.m0 = f("bio_gal"), np.asarray(z["bio_m"]).astype(int)
        self.tau, self.width, self.t0 = float(z["bio_tau"]), float(z["bio_ramp"]), int(z["bio_t0"])
        self.d = self.V.shape[1]
        self._c = {}

    def proj(self, F):
        return ((np.asarray(F, np.float64) - self.mu) / self.sd) @ self.V

    def blocks(self, n, m):
        k = (n, m)
        if k not in self._c:
            d = self.d
            A, C = self.B + self.W / m, self.B + self.W + self.S / n
            same = np.block([[A, self.B], [self.B, C]])
            M = np.linalg.inv(same)
            ld = np.linalg.slogdet(same)[1]
            self._c[k] = (M[:d, :d], M[:d, d:], M[d:, d:], np.linalg.inv(A), np.linalg.inv(C), np.linalg.slogdet(A)[1], np.linalg.slogdet(C)[1], ld)
        return self._c[k]

    def score(self, g, m, x, n):
        """g (M,d) gallery means with session counts m (M,), x (K,d) probe means over n windows -> (K, M) per-dim LLR (package formula)."""
        out = np.empty((len(x), len(g)))
        for mm in np.unique(m):
            M11, M12, M22, Ai, Ci, ldA, ldC, ld = self.blocks(n, int(mm))
            ix = np.flatnonzero(m == mm)
            gi = g[ix]
            gAg = np.einsum("id,de,ie->i", gi, Ai, gi)
            xCx = np.einsum("kd,de,ke->k", x, Ci, x)
            q = (np.einsum("id,de,ie->i", gi, M11, gi)[None] + 2 * np.einsum("id,de,ke->ki", gi, M12, x)
                 + np.einsum("kd,de,ke->k", x, M22, x)[:, None])
            out[:, ix] = (-0.5 * (q + ld) + 0.5 * (gAg[None] + xCx[:, None] + ldA + ldC)) / self.d
        return out

    def stream(self, g, m, Fp):
        """Fp (T,d) projected per-window features in delivery order -> (T, M) scores for n = 1..T (prefix means)."""
        cs = np.cumsum(Fp, 0)
        return np.stack([self.score(g, m, cs[n - 1:n] / n, n)[0] for n in range(1, len(Fp) + 1)])
