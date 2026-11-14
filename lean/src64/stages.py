"""Session stages after the member mix: rebalancing, SessionEM and TSEM, chained as a Tail."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .numerics import SFREQ, lda_posterior


class Rebalance:
    """Prior correction: the mix divided by m ** alpha, m being the mean mix of the session's last W windows
    blended with a uniform prior that counts as k windows."""

    def __init__(self, z):
        self.alpha, self.k, self.W = float(z["rb_alpha"]), float(z["rb_k"]), int(z["rb_w"])
        self.reset_state()

    def reset_state(self):
        self.hist = []

    def step(self, p):
        h = self.hist[-self.W:]
        m = (self.k / len(p) + np.asarray(h).sum(0)) / (self.k + len(h))
        q = p / m ** self.alpha
        self.hist = (self.hist + [p])[-self.W:]
        return q / q.sum()


def early_boost(boost, T, t, gate):
    """Factor on a stage's exponent: 1 + boost * gate up to window T, falling linearly to 1 at window 2 T."""
    return 1 + boost * (min(max((2 * T - t) / T, 0.0), 1.0) * gate)


class SessionEM:
    """Per-session class templates in a view of frontal slow potentials and ocular-event rates: window t is
    reweighted by Gaussian templates fitted on the windows before it. ``gate`` (for the early boost, also used by
    TSEM) is 1 for a session with low mean input margin so far, 0 for a confident one."""

    def __init__(self, z):
        self.chans = [z[k].astype(int) for k in ("em_hl", "em_hr", "em_vt", "em_vb")]
        self.lam, self.n0, self.m0 = float(z["em_lam"]), float(z["em_n0"]), float(z["em_m0"])
        self.shrink, self.iters, self.nmin = float(z["em_shrink"]), int(z["em_iters"]), int(z["em_nmin"])
        self.bins = int(z["em_bins"])
        self.ev_v = [z[k].astype(int) for k in ("em_ev_vt", "em_ev_vb")]
        self.ev_g, self.ev_k = z["em_ev_g"].astype(np.float64), float(z["em_ev_k"])
        self.ev_q, self.ev_bins = z["em_ev_q"].astype(np.float64), int(z["em_ev_bins"])
        self.ev_sos = butter(4, 8, btype="low", fs=SFREQ, output="sos")
        self.boost, self.boost_T = float(z["em_boost"]), float(z["eb_T"])
        self.g_lo, self.g_hi = float(z["eb_gate_lo"]), float(z["eb_gate_hi"])
        self.reset_state()

    def reset_state(self):
        self.A, self.B, self.ev_hist, self.msum, self.gate = [], [], [], 0.0, 1.0
        self.state = None  # v6.4: (m, sd, mu, Ci, e) the last window was reweighted with; None if it was not

    def events(self, x):
        """Log rates of fast horizontal / vertical frontal deflections above multiples of the session's median."""
        n = (x.shape[-1] - 1) // self.ev_bins * self.ev_bins
        h = x[self.chans[0]].mean(0) - x[self.chans[1]].mean(0)
        v = x[self.ev_v[0]].mean(0) - x[self.ev_v[1]].mean(0)
        a = np.abs(np.stack([np.diff(sosfiltfilt(self.ev_sos, s))[:n] for s in (h, v)]))
        self.ev_hist.append(np.median(a, axis=1))
        m = (self.ev_k * self.ev_g + np.sum(self.ev_hist, 0)) / (self.ev_k + len(self.ev_hist))
        f = [(a[i] > q * m[i]).reshape(self.ev_bins, -1).mean(-1) for i in range(2) for q in self.ev_q]
        return np.log(np.concatenate(f) + 1e-3)

    def features(self, x):
        x = x.astype(np.float64)
        t = x.shape[-1] // self.bins * self.bins
        hl, hr, vt, vb = (x[c].mean(0)[:t].reshape(self.bins, -1).mean(-1) for c in self.chans)
        return np.concatenate([hl - hr, vt - vb, self.events(x)])

    def step(self, x, p):
        f, t, out = self.features(x), len(self.A), p
        self.gate = 1.0 if t == 0 else min(max((self.g_hi - self.msum / t) / (self.g_hi - self.g_lo), 0.0), 1.0)
        self.state = None
        if t >= self.nmin:
            try:
                out = self._adapt(np.asarray(self.A), np.asarray(self.B), f, p, t)
            except np.linalg.LinAlgError:  # degenerate view (e.g. flat frontal channels)
                pass
            if not np.isfinite(out).all():
                out, self.state = p, None
        self.A.append(f)
        self.B.append(p)
        s = np.sort(p)
        self.msum += s[-1] - s[-2]
        return out

    def _adapt(self, A, B, f, p, t):
        m, sd = A.mean(0), np.maximum(A.std(0), 1e-3)  # sd floor: a constant column must not blow up z scores
        Z, z = (A - m) / sd, (f - m) / sd
        Q = B
        for _ in range(self.iters):
            mu = (Q / (Q.sum(0, keepdims=True) + self.m0)).T @ Z
            C = np.cov((Z - Q @ mu).T)
            C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
            Ci = np.linalg.inv(C)
            Q = B * lda_posterior(Z, mu, Ci) ** self.lam
            Q = Q / Q.sum(1, keepdims=True)
        e = self.lam * t / (t + self.n0) * early_boost(self.boost, self.boost_T, t, self.gate)
        self.state = (m, sd, mu, Ci, e)
        q = p * lda_posterior(z[None], mu, Ci)[0] ** e
        return q / q.sum()

    def reapply(self, t):
        """v6.4 r15: windows 0..t-1 under the state window t was reweighted with (their inputs as they entered this
        stage; unchanged where the stage did not adapt window t), each normalised. Called after window t's step."""
        B = np.asarray(self.B[:t])
        if self.state is None:
            return B
        m, sd, mu, Ci, e = self.state
        q = B * lda_posterior((np.asarray(self.A[:t]) - m) / sd, mu, Ci) ** e
        return q / q.sum(1, keepdims=True)


class TSEM:
    """Per-session class templates in the leading PCs of core1's features: per class the most confident earlier
    windows (by the posterior entering this stage) give a mean, their pooled covariance an LDA likelihood."""

    def __init__(self, z):
        self.lam, self.rho, self.shrink = (float(z[f"tsem_{k}"]) for k in ("lam", "rho", "shrink"))
        self.m0, self.n0, self.nmin = float(z["tsem_m0"]), float(z["tsem_n0"]), int(z["tsem_nmin"])
        self.boost, self.boost_T = float(z["tsem_boost"]), float(z["eb_T"])
        self.reset_state()

    def reset_state(self):
        self.U, self.conf, self.lab = [], [], []
        self.state = None  # v6.4: (m, sd, mu, Cinv, e) the last window was reweighted with; None if it was not

    def step(self, u, q, gate):
        t, out = len(self.U), q
        self.state = None
        if t >= self.nmin:
            try:
                out = self._adapt(np.asarray(self.U), np.asarray(self.conf), np.asarray(self.lab), u, q, t, gate)
            except np.linalg.LinAlgError:
                out, self.state = q, None
            if not np.isfinite(out).all():
                out, self.state = q, None
        self.U.append(u)
        self.conf.append(q.max())
        self.lab.append(q.argmax())
        return out

    def _adapt(self, A, conf, lab, u, q, t, gate):
        m, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
        Z, z = (A - m) / sd, (u - m) / sd
        k = max(2, int(self.rho * t / 3))
        sel = [i[np.argsort(-conf[i])[:k]] for i in (np.flatnonzero(lab == c) for c in range(3))]
        if min(len(i) for i in sel) < 2:
            return q
        mu = np.stack([Z[i].sum(0) / (len(i) + self.m0) for i in sel])
        C = np.cov(np.concatenate([Z[i] - mu[c] for c, i in enumerate(sel)]).T)
        C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
        e = self.lam * t / (t + self.n0) * early_boost(self.boost, self.boost_T, t, gate)
        Ci = np.linalg.inv(C)
        self.state = (m, sd, mu, Ci, e)
        qq = q * lda_posterior(z[None], mu, Ci)[0] ** e
        return qq / qq.sum()

    def reapply(self, t, Q):
        """v6.4 r15: Q (windows 0..t-1, as relabelled by SessionEM) under the state window t was reweighted with."""
        if self.state is None:
            return Q
        m, sd, mu, Ci, e = self.state
        q = Q * lda_posterior((np.asarray(self.U[:t]) - m) / sd, mu, Ci) ** e
        return q / q.sum(1, keepdims=True)


class Tail:
    """Rebalancing -> SessionEM -> TSEM on one stream of mixes. The model runs two tails: one on the plain mix
    (it feeds the gates) and one on the routed mix (it gives the output)."""

    def __init__(self, z):
        self.rb, self.em, self.tsem = Rebalance(z), SessionEM(z), TSEM(z)

    def reset_state(self):
        for stage in (self.rb, self.em, self.tsem):
            stage.reset_state()

    def reapply(self, t):
        """v6.4 r15: windows 0..t-1 under SessionEM's and TSEM's state for window t (ledger s221 / s229, relabel.py)."""
        return self.tsem.reapply(t, self.em.reapply(t))

    def step(self, x, u, mix):
        """Raw window, TSEM view of the window, mix -> posterior."""
        q = self.em.step(x, self.rb.step(mix))
        return self.tsem.step(u, q, self.em.gate)
