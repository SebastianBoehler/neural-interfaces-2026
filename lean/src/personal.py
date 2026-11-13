"""Own-space personal stage: models fitted on a public person's sessions in that person's own space."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .numerics import SFREQ, softmax


class OwnStage:
    """Three members per public person: T = tangent space of ``own_bands`` covariances recentred by the session's
    running mean covariance (the person's public mean counts as ``kref`` windows), logistic regression; C = FBCSP
    log-variances of the same recentred covariances, LDA; S = slow time courses (below 4 Hz, 8 per second,
    standardised over the session so far), LDA. ``observe`` runs on every window; ``apply`` multiplies a routed
    posterior by exp(beta * ramp * mean clipped log posterior of T, C, S)."""

    def __init__(self, z):
        f = lambda k: np.asarray(z[k], dtype=np.float64)
        self.beta, self.kref = float(z["own_beta"]), float(z["own_kref"])
        self.sos = [butter(4, [float(lo), float(hi)], btype="band", fs=SFREQ, output="sos")
                    for lo, hi in f("own_bands")]
        self.sos_slow = butter(2, 4, btype="low", fs=SFREQ, output="sos")
        self.prior, self.csp = f("own_prior"), f("own_csp")  # (people, bands, C, C), (people, bands, C, filters)
        self.t_mu, self.t_sd, self.t_W, self.t_c = f("own_t_mu"), f("own_t_sd"), f("own_t_W"), f("own_t_c")
        self.c_W, self.c_c, self.s_W, self.s_c = f("own_c_W"), f("own_c_c"), f("own_s_W"), f("own_s_c")
        n = self.prior.shape[-1]
        self.iu = np.triu_indices(n)
        self.wt = np.where(self.iu[0] == self.iu[1], 1.0, np.sqrt(2.0))
        self.eye = np.eye(n) / n
        self.reset_state()

    def reset_state(self):
        self.n, self.csum, self.s1, self.s2, self.cur = 0, 0.0, 0.0, 0.0, None

    def observe(self, x):
        """Add the window to the session sums. False for a degenerate window (non-finite or a flat channel): it is
        neither added nor staged."""
        x = np.asarray(x, dtype=np.float64)
        self.cur = None
        sd = x.std(1)
        if not (np.isfinite(x).all() and sd.min() > 1e-9 * max(float(sd.max()), 1e-30)):
            return False
        xc = x - x.mean(0, keepdims=True)
        C = []
        for sos in self.sos:
            b = sosfiltfilt(sos, xc, axis=-1)
            c = b @ b.T / b.shape[-1]
            C.append(c + 1e-3 * np.trace(c) * self.eye)
        C = np.stack(C)
        sl = sosfiltfilt(self.sos_slow, x, axis=-1)[:, 7::15].ravel()
        self.csum, self.s1, self.s2 = self.csum + C, self.s1 + sl, self.s2 + sl ** 2
        self.n += 1
        self.cur = (C, sl)
        return True

    def log_members(self, j):
        """Mean clipped log posterior of person j's three members for the window last observed."""
        C, sl = self.cur
        n = self.n
        w, V = np.linalg.eigh((self.kref * self.prior[j] + self.csum) / (self.kref + n))
        W = np.einsum("bij,bj,bkj->bik", V, 1 / np.sqrt(w), V)
        Cw = W @ C @ W
        w, V = np.linalg.eigh(Cw)
        Lg = np.einsum("bij,bj,bkj->bik", V, np.log(w), V)
        t = (Lg[:, self.iu[0], self.iu[1]] * self.wt).ravel()
        pt = softmax(((t - self.t_mu[j]) / self.t_sd[j])[None] @ self.t_W[j].T + self.t_c[j])[0]
        fc = np.log(np.einsum("bcf,bcd,bdf->bf", self.csp[j], Cw, self.csp[j])).ravel()
        pc = softmax(fc[None] @ self.c_W[j].T + self.c_c[j])[0]
        m = self.s1 / n
        z = (sl - m) / np.sqrt(np.maximum(self.s2 / n - m ** 2, 1e-9)) if n >= 4 else np.zeros_like(sl)
        ps = softmax(z[None] @ self.s_W[j].T + self.s_c[j])[0]
        return sum(np.log(np.clip(q, 1e-3, 1)) for q in (pt, pc, ps)) / 3

    def apply(self, out, j, r):
        """(staged posterior, log members) for a window routed to person j with ramp r, or None if not finite."""
        lm = self.log_members(j)
        o = np.log(np.clip(out, 1e-9, 1)) + self.beta * r * lm
        o = np.exp(o - o.max())
        o = o / o.sum()
        return (o, lm) if np.isfinite(lm).all() and np.isfinite(o).all() else None
