"""Reuse the current stock count view while keeping separate native urn histories."""

import numpy as np

from combined_selector import SharedCovRetiringSelector


class SharedViewRetiringSelector(SharedCovRetiringSelector):
    def __init__(self, model, audit=False):
        super().__init__(model, audit)
        stock, alternate = model.cf, self.outer.cf
        if len(stock.sos) != len(alternate.sos) or stock.V is alternate.V or np.shares_memory(stock.al, alternate.al):
            raise ValueError("the two native count filters do not have independent state")
        np.testing.assert_array_equal(stock.W, alternate.W)
        np.testing.assert_array_equal(stock.sw, alternate.sw)
        for left, right in zip(stock.sos, alternate.sos, strict=True):
            np.testing.assert_array_equal(left, right)
        self.independent_count_view = alternate.view
        alternate.view = self.current_count_view
        self.count_view_calls = 0
        self.verified_count_views = 0

    def current_count_view(self, x):
        stock, alternate = self.model.cf, self.outer.cf
        i = self.age
        if (self.frame is None or self.frame[-1] or stock.dead or alternate.dead
                or stock.t != i + 1 or alternate.t != i or len(stock.V) != i + 1 or len(alternate.V) != i
                or not np.array_equal(x, self.frame[0])):
            raise ValueError("count reuse requires the exact completed stock view for this raw window")
        row = stock.V[i]
        own = row.copy()
        if self.audit:
            np.testing.assert_array_equal(own, self.independent_count_view(x))
            if np.shares_memory(own, row):
                raise ValueError("alternate count view must own its copy")
            self.verified_count_views += 1
        self.count_view_calls += 1
        return own

    def summary(self):
        proof = super().summary()
        n = proof["alternate_rows"]
        stock, alternate = self.model.cf, self.outer.cf
        expected_latch = stock.latch if stock.latch is not None and stock.latch < n else None
        if (self.count_view_calls != n or self.verified_count_views != (n if self.audit else 0)
                or len(alternate.V) != n or alternate.t != n or alternate.latch != expected_latch
                or stock.t != self.age or len(stock.V) != self.age):
            raise ValueError("native count view reuse or its independent latch was bypassed")
        return dict(proof, count_view_calls=self.count_view_calls, verified_count_views=self.verified_count_views,
                    stock_count_latch=stock.latch, alternate_count_latch=alternate.latch)
