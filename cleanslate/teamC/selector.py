"""Session-local decisions from two whole decoder streams and past EEG."""

import numpy as np

RULES = ("nearest", "shuffled", "confidence", "prior", "consistency")
SEED = 20261120


def tangent(x):
    """Logarithm of a trace-normalized, shrinkage 8–30 Hz covariance."""
    freq = np.fft.rfftfreq(x.shape[1], d=1 / 120)
    bands = np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)[:, (freq >= 8) & (freq <= 30)]
    cov = (bands @ bands.conj().T).real
    trace = np.trace(cov)
    if not np.isfinite(trace) or trace <= 0:
        raise ValueError("invalid band covariance")
    cov *= 0.9 / trace
    cov.flat[::len(cov) + 1] += 0.1 / len(cov)
    eig, vec = np.linalg.eigh(cov)
    return ((vec * np.log(eig)) @ vec.T).ravel()


class Selector:
    def __init__(self):
        self.x, self.cap, self.team, self.margin, self.agree = [], [], [], [], []
        self.cap_count = np.zeros(3, dtype=int)
        self.team_count = np.zeros(3, dtype=int)
        self.rng = np.random.default_rng(SEED)

    def step(self, x, cap, team, post, enabled=True):
        if not enabled:
            return np.full(len(RULES), cap, dtype=np.int8)
        feat = tangent(x)
        p = np.sort(post)
        margin = p[-1] - p[-2]
        votes = np.zeros(len(RULES), dtype=bool)
        if cap != team:
            votes = self.support(feat, cap, team, margin)
        if cap == team:
            self.agree.append(len(self.x))
        self.x.append(feat)
        self.cap.append(cap)
        self.team.append(team)
        self.margin.append(margin)
        self.cap_count[cap] += 1
        self.team_count[team] += 1
        return np.where(votes, team, cap).astype(np.int8)

    def support(self, feat, cap, team, margin):
        n = len(self.x)
        agreement = np.asarray(self.agree, dtype=int)
        cap_history = np.asarray(self.cap)
        team_history = np.asarray(self.team)
        nearest = shuffled = consistency = False
        if n >= 5:
            diff = np.asarray(self.x) - feat
            distance = np.einsum("ij,ij->i", diff, diff)
            close = np.argsort(distance, kind="stable")[:5]
            team_support = np.count_nonzero(cap_history[close] == team)
            consistency = team_support >= 3 and team_support > np.count_nonzero(team_history[close] == cap)
            if len(agreement) >= 5:
                neighbours = np.argsort(distance[agreement], kind="stable")[:5]
                labels = cap_history[agreement]
                nearest = np.count_nonzero(labels[neighbours] == team) >= 3
                shuffled = np.count_nonzero(self.rng.permutation(labels)[neighbours] == team) >= 3
        confidence = (len(agreement) >= 12 and len(agreement) / n >= 0.50
                      and margin >= np.quantile(np.asarray(self.margin)[agreement], 0.75))
        # Both candidate histograms have the same total, so sum of squared counts
        # orders their squared distances from uniform.
        prior = n >= 15 and (
            self.team_count @ self.team_count + 2 * self.team_count[team]
            < self.cap_count @ self.cap_count + 2 * self.cap_count[cap]
        )
        return nearest, shuffled, confidence, prior, consistency
