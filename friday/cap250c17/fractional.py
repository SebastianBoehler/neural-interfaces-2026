"""Query two fractional histories through the actual cap250 alignment stream."""

import math

import numpy as np

from c12_history import History, digest, target_group


class FractionalEA:
    def __init__(self, source, mode):
        assert mode in ("disabled", "informative", "permuted")
        self.source, self.mode = source, mode
        assert source.G.shape == (41, 41) and source.G.dtype == np.float64
        self.prior = digest(source.G)
        self.trace = float(np.trace(source.G))
        self.prior_norm = float(np.linalg.norm(source.G))
        assert math.isfinite(self.trace) and self.trace > 0
        self.reset_audit()

    def reset_audit(self):
        self.total = 0.
        self.high = np.zeros((41, 41))
        self.pending = []
        self.mass = self.separation = 0.
        self.weights, self.branch_mass = [], []
        self.groups = self.moved = 0

    def reset_state(self):
        self.source.reset_state()
        self.reset_audit()

    def select(self, age, u):
        if age == 0:
            return self.total
        prior_hi = self.high.copy()
        for c, w in self.pending:
            prior_hi += w*c
        prior_lo = self.total-prior_hi
        fit = self.source.k*self.source.G
        hi = (fit+prior_hi)/(self.source.k+self.mass)
        lo = (fit+prior_lo)/(self.source.k+age-self.mass)
        separation = float(np.linalg.norm(hi-lo)/self.prior_norm)
        self.separation = max(self.separation, separation)
        exposure = u*self.mass+(1-u)*(age-self.mass)
        assert 0 < exposure <= age
        return (u*prior_hi+(1-u)*prior_lo)*(age/exposure)

    def commit(self, cov, u):
        self.total = self.total+cov
        self.mass += u
        self.pending.append((cov,u))
        if len(self.pending) == 4:
            weights = [w for _, w in self.pending]
            assigned = weights if self.mode == "informative" else weights[2:]+weights[:2]
            assert sorted(weights) == sorted(assigned) and math.fsum(weights) == math.fsum(assigned)
            for (c, _), w in zip(self.pending, assigned, strict=True):
                self.high += w*c
            self.pending.clear()
            self.groups += 1
            self.moved += 4*int(self.mode == "permuted")

    def __call__(self, x):
        age = self.source.n
        if self.mode == "disabled":
            result = self.source(x)
            assert self.source.n == age+1
            return result
        assert x.shape == (41,480) and x.dtype == np.float32 and np.isfinite(x).all()
        assert self.source.n == len(self.weights) and np.array_equal(self.source.cov_sum, self.total)
        original = x.copy()
        C = History.increment(self, x)
        ratio = float(np.trace(C))/self.trace
        assert ratio > 0 and math.isfinite(ratio)
        u = .5+.4*math.tanh(math.log2(ratio))
        assert .09999999 <= u <= .90000001
        self.weights.append(u)
        self.branch_mass.append(self.mass)
        chosen = self.select(age,u)
        self.source.cov_sum = chosen
        # The exact package method gives the current increment weight one.
        result = self.source(x)
        assert self.source.n == age+1 and np.array_equal(self.source.cov_sum, chosen+C)
        assert np.array_equal(x, original) and np.isfinite(result).all()
        self.commit(C,u)
        self.source.cov_sum = self.total
        return result

    def receipt(self):
        assert digest(self.source.G) == self.prior
        age = self.source.n
        if self.mode != "disabled":
            assert (len(self.weights), self.groups, len(self.pending)) == (age, age//4, age%4)
            assert len(self.branch_mass) == age and np.array_equal(self.source.cov_sum, self.total)
        return dict(mode=self.mode, rows=age, prior_sha=self.prior, weights=self.weights,
                    branch_mass=self.branch_mass, separation=self.separation, moved=self.moved, blocks=self.groups)


def install(model, mode):
    i, source, members = target_group(model)
    if mode == "stock":
        return source
    split = FractionalEA(source, mode)
    model.nets.groups[i] = (split, members)
    return split
