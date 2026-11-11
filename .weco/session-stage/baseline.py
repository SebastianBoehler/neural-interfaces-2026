"""Label-free, causal session adaptation for a 3-class EEG decoder (motor imagery, mental subtraction, word generation).

A fixed ensemble has already produced, for every 4-second window of one recording session, a 3-class probability
vector. This stage sees the session as a stream and may adjust each window's probabilities using only that window and
earlier windows of the same session (no labels, no future windows). The final label is the argmax of the output.

Known structure of a session (about 190 windows): it consists of runs of 30-36 windows, and inside every run the three
classes occur equally often in a shuffled order. The class-conditional feature patterns differ from person to person,
which is why per-session templates help.
"""
import numpy as np


def _lda(Z, mu, Ci):
    """Row-wise class likelihoods (softmax of LDA scores) for class means mu and inverse covariance Ci."""
    W = Ci @ mu.T
    s = Z @ W - 0.5 * (mu * W.T).sum(1)
    s = s - s.max(-1, keepdims=True)
    g = np.exp(s)
    return g / g.sum(-1, keepdims=True)


def _early(t, T=30):
    """1 until window T, falling linearly to 0 at window 2T."""
    return float(np.clip((2 * T - t) / T, 0, 1)) if t >= 0 else 0.0


def run_session(P, X, F):
    """P (n, 3): ensemble probabilities per window, in recording order.
    X (n, 128): a 128-dimensional feature view of each window (principal components of covariance features).
    F (n, d): ocular and slow-potential features of each window.
    Returns (n, 3) adjusted probabilities. Row t may depend only on rows 0..t of P, X and F."""
    n = len(P)
    # 1. Rebalancing: divide by the recent class share (last 30 windows, 5 pseudo-windows of a uniform prior).
    alpha, k, W = 1.15, 5.0, 30
    b = np.empty_like(P)
    for t in range(n):
        lo = max(0, t - W)
        m = (k / 3.0 + P[lo:t].sum(0)) / (k + (t - lo))
        q = P[t] / m ** alpha
        b[t] = q / q.sum()
    # 2. Gate for the early boost: sessions whose running mean margin is low get a stronger early adaptation.
    srt = np.sort(b, 1)
    margin = srt[:, -1] - srt[:, -2]
    g = np.ones(n)
    for t in range(1, n):
        g[t] = np.clip((0.35 - margin[:t].mean()) / 0.15, 0.0, 1.0)
    # 3. Session EM on the ocular / slow features: soft class templates from all earlier windows.
    lam, n0, m0, shrink, nmin, boost = 0.5, 60.0, 20.0, 0.7, 20, 2.0
    e = b.copy()
    for t in range(nmin, n):
        try:
            A, B, f, p = F[:t], b[:t], F[t], b[t]
            mean, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
            Z, z = (A - mean) / sd, (f - mean) / sd
            Q = B
            for _ in range(2):
                mu = (Q / (Q.sum(0, keepdims=True) + m0)).T @ Z
                C = np.cov((Z - Q @ mu).T)
                C = (1 - shrink) * C + shrink * np.eye(len(C)) * np.trace(C) / len(C)
                Ci = np.linalg.inv(C)
                Q = B * _lda(Z, mu, Ci) ** lam
                Q = Q / Q.sum(1, keepdims=True)
            ex = lam * t / (t + n0) * (1 + boost * (_early(t) * g[t]))
            q = p * _lda(z[None], mu, Ci)[0] ** ex
            q = q / q.sum()
            if np.isfinite(q).all():
                e[t] = q
        except np.linalg.LinAlgError:
            pass
    # 4. Template stage on the 128-dimensional view: hard class templates from the most confident earlier windows.
    lam2, rho, shrink2, m02, nmin2, boost2 = 0.15, 0.7, 0.7, 20.0, 15, 4.0
    out = e.copy()
    lab, conf = e.argmax(1), e.max(1)
    for t in range(nmin2, n):
        A, f = X[:t], X[t]
        mean, sd = A.mean(0), np.maximum(A.std(0), 1e-3)
        Z, z = (A - mean) / sd, (f - mean) / sd
        quota = max(2, int(rho * t / 3))
        sel = []
        for c in range(3):
            ik = np.flatnonzero(lab[:t] == c)
            sel.append(ik[np.argsort(-conf[ik])[:quota]])
        if min(len(x) for x in sel) < 2:
            continue
        mu = np.stack([Z[x].sum(0) / (len(x) + m02) for x in sel])
        R = np.concatenate([Z[x] - mu[c] for c, x in enumerate(sel)])
        C = np.cov(R.T)
        C = (1 - shrink2) * C + shrink2 * np.eye(len(C)) * np.trace(C) / len(C)
        like = _lda(z[None], mu, np.linalg.inv(C))[0]
        ex = lam2 * t / (t + 60.0) * (1 + boost2 * (_early(t) * g[t]))
        q = e[t] * like ** ex
        out[t] = q / q.sum()
    return out
