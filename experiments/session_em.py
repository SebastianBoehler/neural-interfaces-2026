"""Causal per-session class templates in a low-dimensional ocular view (R19).

The screen layout is the same for everyone, so each task comes with a
consistent pattern of slow frontal potentials (gaze and eye movements), but
its exact form is person-specific. Per session, a Gaussian class template in
this 32-dimensional view is estimated from the session's own past windows:
soft EM, seeded by the ensemble's rebalanced posteriors, with class-balanced
responsibilities, class means shrunk toward the session mean (prior count
``m0``) and a shrunk shared covariance. Its likelihood multiplies the
ensemble posterior as a product of experts, with exponent lam * n / (n + n0).

The view is separate from the classifier, so the template does not just
re-learn the classifier's errors (the confirmation bias that sank
self-training in R4/R8). Window t uses only windows < t for the template and
the standardisation; it starts after ``nmin`` windows.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

FRONT_L, FRONT_R = ("F5", "AF3", "F3"), ("F6", "AF4", "F4")
VERT_TOP, VERT_BOTTOM = ("AF3", "AF4"), ("P1", "P2", "Pz")
EV_TOP, EV_BOTTOM = ("AF3", "AF4"), ("C1", "C2", "Cz")
EV_SOS = butter(4, 8, "low", fs=120, output="sos")


def gaze_features(X, ch_names, bins=16):
    """(N, 41, T) microvolts -> (N, 2 * bins): horizontal and vertical
    frontal slow-potential differences, averaged in ``bins`` time bins."""
    ix = {c: i for i, c in enumerate(ch_names)}
    sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)
    n, t = len(X), X.shape[-1] // bins * bins
    h = (sel(FRONT_L) - sel(FRONT_R))[:, :t].reshape(n, bins, -1).mean(-1)
    v = (sel(VERT_TOP) - sel(VERT_BOTTOM))[:, :t].reshape(n, bins, -1).mean(-1)
    return np.concatenate([h, v], 1).astype(np.float64)


def event_traces(X, ch_names, bins=4):
    """(N, 41, T) -> (2, N, T') horizontal and vertical 8 Hz low-passed first
    differences (saccade and blink velocity proxies), T' divisible by bins."""
    ix = {c: i for i, c in enumerate(ch_names)}
    sel = lambda cs: X[:, [ix[c] for c in cs]].mean(1)
    n = (X.shape[-1] - 1) // bins * bins
    return np.stack([np.diff(sosfiltfilt(EV_SOS, a, axis=-1), axis=-1)[:, :n]
                     for a in (sel(FRONT_L) - sel(FRONT_R), sel(EV_TOP) - sel(EV_BOTTOM))])


def global_mad(X, ch_names):
    """Prior event scale (median |trace| over all training samples), (2,)."""
    return np.median(np.abs(event_traces(X, ch_names)), axis=(1, 2))


class EventRates:
    """Causal per-session ocular-event rates (R19e): log fraction of samples
    per time bin where |trace| exceeds q times the running per-window median
    |trace| of the session so far (including the current window), blended
    with the training prior g weighted as k windows. Sign-free; 2 traces x
    len(q) thresholds x bins values (16)."""

    def __init__(self, g, k=5, q=(4, 8), bins=4):
        self.g, self.k, self.q, self.bins = np.asarray(g, float), k, q, bins
        self.hist = []

    def step(self, tr):  # tr: (2, T') one window's traces
        a = np.abs(tr)
        self.hist.append(np.median(a, axis=1))
        m = (self.k * self.g + np.sum(self.hist, 0)) / (self.k + len(self.hist))
        f = [(a[i] > q * m[i]).reshape(self.bins, -1).mean(-1) for i in range(2) for q in self.q]
        return np.log(np.concatenate(f) + 1e-3)


def event_features(X, ch_names, streams, g, **kw):
    """Offline helper: EventRates over rows in acquisition order, per stream."""
    T = event_traces(X, ch_names)
    out = np.empty((len(X), 2 * 2 * 4))
    for s in dict.fromkeys(streams):
        er = EventRates(g, **kw)
        for r in np.flatnonzero(streams == s):
            out[r] = er.step(T[:, r])
    return out


def _softmax_lda(Z, mu, Ci):
    W = Ci @ mu.T
    s = Z @ W - 0.5 * (mu * W.T).sum(1)
    s -= s.max(-1, keepdims=True)
    g = np.exp(s)
    return g / g.sum(-1, keepdims=True)


class SessionEM:
    """Streaming stage: ``step(feature, posterior) -> adjusted posterior``.
    One instance per session (reset at session boundaries)."""

    def __init__(self, lam=0.5, n0=60, m0=20, shrink=0.7, iters=2, nmin=10):
        self.lam, self.n0, self.m0, self.shrink = lam, n0, m0, shrink
        self.iters, self.nmin = iters, nmin
        self.A, self.B = [], []

    def step(self, f, p):
        t = len(self.A)
        out = p
        if t >= self.nmin:
            A, B = np.asarray(self.A), np.asarray(self.B)
            m, sd = A.mean(0), A.std(0) + 1e-9
            Z, z = (A - m) / sd, (f - m) / sd
            Q = B
            for _ in range(self.iters):
                mu = (Q / (Q.sum(0, keepdims=True) + self.m0)).T @ Z
                C = np.cov((Z - Q @ mu).T)
                C = (1 - self.shrink) * C + self.shrink * np.eye(len(C)) * np.trace(C) / len(C)
                Ci = np.linalg.inv(C)
                Q = B * _softmax_lda(Z, mu, Ci) ** self.lam
                Q = Q / Q.sum(1, keepdims=True)
            e = self.lam * t / (t + self.n0) if self.n0 else self.lam
            q = p * _softmax_lda(z[None], mu, Ci)[0] ** e
            out = q / q.sum()
        self.A.append(f)
        self.B.append(p)
        return out


def apply(F, P, streams, **kw):
    """Run the stage over rows in acquisition order, one SessionEM per stream."""
    out = np.empty_like(P)
    for s in dict.fromkeys(streams):
        em = SessionEM(**kw)
        for r in np.flatnonzero(streams == s):
            out[r] = em.step(F[r], P[r])
    return out
