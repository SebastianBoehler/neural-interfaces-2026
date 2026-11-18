"""Change only what the archived input stream carries into later windows."""
import hashlib

import numpy as np


ALIGNED = tuple([0, 1, 2, 3, 4, 20, 21, *range(34, 47), *range(53, 65)])
CAP = 8.0


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def target_group(model):
    groups = [(i, src, members) for i, (src, members) in enumerate(model.nets.groups) if src.align]
    assert len(groups) == 1
    i, src, members = groups[0]
    assert tuple(j for j, _ in members) == ALIGNED
    assert type(src).__name__ == 'InputStream' and src.n == 0 and src.cov_sum == 0
    assert src.k == 5.0 and src.shrink == .001
    return i, src, members


class History:
    def __init__(self, source, arm):
        assert arm in ('zero', 'directional', 'scalar')
        self.source = source
        self.arm = arm
        prior = source.G.copy()
        assert prior.dtype == np.float64 and prior.shape == (41, 41) and np.isfinite(prior).all()
        np.testing.assert_allclose(prior, prior.T, atol=1e-12*np.abs(prior).max(), rtol=0)
        eigen, axes = np.linalg.eigh(prior)
        self.prior_ratio = float(eigen[0]/eigen[-1])
        assert eigen[-1] > 0 and self.prior_ratio > 1e-10
        self.root = (axes*np.sqrt(eigen))@axes.T
        self.white = (axes/np.sqrt(eigen))@axes.T
        self.prior_sha256 = digest(prior)
        self.reset_audit()

    def reset_audit(self):
        self.ages, self.traces = [], []
        self.max_cov_difference = self.max_trace_error = self.max_eigen = 0.

    def reset_state(self):
        self.source.reset_state()
        self.reset_audit()

    def increment(self, x):
        # Same order, dtype and population normalization as the archived input stream.
        xc = x-x.mean(-1, keepdims=True)
        cov = np.einsum('nct,ndt->ncd', xc[None], xc[None], dtype=np.float64)/x.shape[-1]
        tr = np.trace(cov, axis1=1, axis2=2)[:, None, None]/cov.shape[1]
        return ((1-self.source.shrink)*cov+self.source.shrink*tr*np.eye(cov.shape[1]))[0]

    def retain(self, cov):
        white = self.white@cov@self.white.T
        eigen, axes = np.linalg.eigh((white+white.T)*.5)
        assert np.isfinite(eigen).all()
        self.max_eigen = max(self.max_eigen, float(eigen[-1]))
        mask = eigen > CAP
        if not mask.any():
            return None
        original = float(np.trace(cov))
        directions = self.root@axes[:, mask]
        drop = (directions*(eigen[mask]-CAP))@directions.T
        directional = cov-drop
        trace = float(np.trace(directional))
        assert 0 < trace < original
        scalar = cov*(trace/original)
        error = abs(float(np.trace(scalar))-trace)/trace
        assert error <= 1e-12
        self.max_trace_error = max(self.max_trace_error, error)
        difference = float(np.max(np.abs(directional-scalar)))
        self.max_cov_difference = max(self.max_cov_difference, difference)
        if not self.ages:
            d = self.white@directional@self.white.T
            s = self.white@scalar@self.white.T
            eig = np.linalg.eigvalsh((d+d.T)*.5)
            assert eig[0] > 0 and abs(eig[-1]-CAP) < 1e-8
            assert np.linalg.eigvalsh((s+s.T)*.5)[0] > 0
        self.traces.append(trace)
        return directional if self.arm == 'directional' else scalar

    def __call__(self, x):
        assert x.shape == (41, 480) and x.dtype == np.float32 and np.isfinite(x).all()
        age, before = self.source.n, self.source.cov_sum
        current = x.copy()
        result = self.source(x)
        assert self.source.n == age+1 and np.array_equal(current, x)
        if self.arm == 'zero':
            return result
        cov = self.increment(x)
        assert np.array_equal(self.source.cov_sum, before+cov)
        retained = self.retain(cov)
        if retained is not None:
            self.source.cov_sum = before+retained
            self.ages.append(age)
        return result

    def receipt(self):
        assert digest(self.source.G) == self.prior_sha256
        return dict(windows=self.source.n, prior=self.prior_sha256, prior_eigen_ratio=self.prior_ratio,
                    active_ages=self.ages, retained_traces=self.traces, max_prior_whitened_eigen=self.max_eigen,
                    scalar_trace_max_relative_error=self.max_trace_error, directional_scalar_cov_max=self.max_cov_difference)


def install(model, arm):
    i, src, members = target_group(model)
    if arm == 'stock':
        return None
    adapter = History(src, arm)
    model.nets.groups[i] = (adapter, members)
    return adapter
