"""Supply the alternate CSP with current, independently owned native covariances."""
import numpy as np

from live_gate import LiveSelector


class SharedCovSelector(LiveSelector):
    def __init__(self, model, audit=False):
        super().__init__(model, audit)
        stock, alternate = model.cp, self.outer.cp
        if len(stock.sos) != len(alternate.sos):
            raise ValueError("private CSP filter counts differ")
        for left, right in zip(stock.sos, alternate.sos, strict=True):
            np.testing.assert_array_equal(left, right)
        self.independent_features = alternate.feats
        alternate.feats = self.current_covariance
        self.shared_calls = 0
        self.verified_covariances = 0

    def current_covariance(self, x):
        stock, alternate = self.model.cp, self.outer.cp
        i = self.age
        if (
            self.frame is None or self.frame[-1] or stock.dead or stock.stopped is not None
            or stock.t != i + 1 or alternate.t != i + 1
            or len(stock.C) != i + 1 or len(alternate.C) != i
            or not np.array_equal(x, self.frame[0])
        ):
            raise ValueError("CSP reuse requires this exact completed stock covariance row")
        owned = stock.C[i].copy()
        if self.audit:
            np.testing.assert_array_equal(owned, self.independent_features(x))
            if np.shares_memory(owned, stock.C[i]):
                raise ValueError("private covariance ownership overlaps")
            self.verified_covariances += 1
        self.shared_calls += 1
        return owned

    def summary(self):
        proof = super().summary()
        expected = self.age if self.recognition_age == -1 else self.recognition_age
        if self.shared_calls != expected or self.verified_covariances != (expected if self.audit else 0):
            raise ValueError("stock covariance reuse was bypassed")
        return dict(proof, shared_calls=self.shared_calls, verified_covariances=self.verified_covariances)
