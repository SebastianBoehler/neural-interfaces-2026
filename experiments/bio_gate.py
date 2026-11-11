"""Label-free identity gate for returning-person routing (v4). Streaming class `Bio` (numpy only; it is pasted verbatim
into submission.py) and `fold_params` to build its parameters from the external model and a set of gallery sessions.

Feature of one window: common-average-referenced log power spectrum 2-40 Hz (1 Hz bins, 7 Hann segments), with each
channel's mean over frequency removed (gain-free spectral shape), 41 x 39 values. The session signature is the running
mean. Score: same-person log-likelihood ratio under a two-covariance Gaussian model (between-person B, within-person
across sessions W, window noise S / n) fitted on external multi-session corpora (experiments/bio_id.py).
"""
import numpy as np


class Bio:
    """step(x) -> (person index, ramp r in [0, 1]); r = clip((score - tau) / ramp, 0, 1) from window t0 on."""

    def __init__(self, p):
        f = lambda k: np.asarray(p[k], dtype=np.float64)  # noqa: E731
        self.mu, self.sd, self.V, self.B, self.W, self.S = (f(k) for k in ("bio_mu", "bio_sd", "bio_V", "bio_B", "bio_W", "bio_S"))
        self.G, self.m = f("bio_gal"), np.asarray(p["bio_m"]).astype(int)
        self.tau, self.ramp, self.t0 = float(p["bio_tau"]), float(p["bio_ramp"]), int(p["bio_t0"])
        self.win, self.d = np.hanning(120), self.V.shape[1]
        self.A = {m: self.B + self.W / m for m in np.unique(self.m)}
        self.gAg = np.array([g @ np.linalg.solve(self.A[m], g) for g, m in zip(self.G, self.m)])
        self.ldA = {m: np.linalg.slogdet(A)[1] for m, A in self.A.items()}
        self.reset_state()

    def reset_state(self):
        self.sum, self.n, self.last = 0.0, 0, None

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
        self.last = None
        if self.n - 1 < self.t0:
            return -1, 0.0
        s = self.scores()
        if not np.isfinite(s).all():
            return -1, 0.0
        j = int(np.argmax(s))
        r = float(np.clip((s[j] - self.tau) / self.ramp, 0, 1))
        self.last = (j, float(s[j]), r)
        return j, r


def fold_params(model, feats, stream, names, tau=0.23, ramp=0.05, t0=10):
    """Parameters for `Bio`: model = arrays sd, V, B, W, S (bio_id export); feats (n, 1599) window features and
    stream ids ("P04|1") of the gallery rows; names = gallery people in the order of the personal heads."""
    sig = {u: feats[stream == u].mean(0) for u in dict.fromkeys(stream)}
    mu = np.mean(list(sig.values()), 0); gal, ms = [], []
    for p in names:
        ss = [v for u, v in sig.items() if u.startswith(p + "|")]
        gal.append(((np.mean(ss, 0) - mu) / model["sd"]) @ model["V"]); ms.append(len(ss))
    return dict(bio_mu=mu, bio_sd=model["sd"], bio_V=model["V"], bio_B=model["B"], bio_W=model["W"], bio_S=model["S"],
                bio_gal=np.stack(gal), bio_m=np.array(ms), bio_tau=tau, bio_ramp=ramp, bio_t0=t0)
