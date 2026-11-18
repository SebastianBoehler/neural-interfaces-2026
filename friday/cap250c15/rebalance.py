"""Bound correction direction in the actual emitted outer-tail rebalancer."""

import math

import numpy as np


CAP = math.log(4)
EXPOSURE_TOL = 2e-12


def centered_log(p):
    log = np.log(p)
    return log - log.mean(axis=-1, keepdims=True)


def corrections(hist, k):
    assert hist.ndim == 2 and hist.shape[1] == 3 and np.isfinite(hist).all() and (hist > 0).all()
    n = len(hist)
    assert 0 < n <= 30
    stock = centered_log((k/3 + hist.sum(0)) / (k+n))
    rows = centered_log(hist)
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    clipped = norms > CAP
    rows *= CAP / np.maximum(CAP, norms)
    robust = rows.sum(axis=0) / (k+n)
    a, b = np.linalg.norm(stock), np.linalg.norm(robust)
    limit = min(a, b)
    robust = robust * (limit/b) if b else robust
    scalar = stock * (limit/a) if a else stock
    error = float(abs(np.linalg.norm(scalar) - np.linalg.norm(robust)))
    assert error <= EXPOSURE_TOL and limit <= a
    return robust, scalar, int(clipped.sum()), error


class Outer:
    """Observe the stock stage and optionally replace only its emitted correction."""

    def __init__(self, model, arm):
        self.arm = arm
        self.tail = model.routeB.tail
        self.rb = self.tail.rb
        assert type(self.tail).__name__ == "TailX" and type(self.rb).__name__ == "Rebalance"
        assert (self.rb.W, self.rb.k, self.rb.alpha, len(self.rb.hist)) == (30, 5., 1.15, 0)
        assert model.routed == 1 and model.tail6r is None
        assert len({id(self.rb), id(model.tail6.rb), id(model.shadow.rb), id(model.route.tail.rb)}) == 4
        self.native_rb, self.native_em, self.native_tail = self.rb.step, self.tail.em.step, self.tail.step
        self.tail.step, self.tail.em.step = self.observe_input, self.observe_em
        if arm != "stock":
            self.rb.step = self.replace
        self.calls = self.em_calls = self.rb_calls = self.changed = self.clip_rows = 0
        self.exposure_error = self.max_scalar_delta = 0.
        self.max_posterior_move = self.max_scalar_posterior_move = 0.
        self.x = self.em_in = self.forwarded = None

    def observe_input(self, *args):
        self.x = args[2].copy()
        self.calls += 1
        snap = self.rb.hist
        expected = snap[-self.rb.W:]
        self.em_in = self.forwarded = None
        result = self.native_tail(*args)
        assert len(self.rb.hist) == min(self.calls, self.rb.W)
        assert np.array_equal(self.rb.hist[-1], self.x)
        assert len(expected) == min(self.rb.W, self.calls-1)
        assert all(old is now for old, now in zip(expected[-29:], self.rb.hist[:-1], strict=True))
        assert self.em_calls == self.calls
        assert self.arm == "stock" or self.rb_calls == self.calls
        return result

    def observe_em(self, x, p):
        self.em_in = p.copy()
        self.em_calls += 1
        if self.arm != "stock":
            assert np.array_equal(p, self.forwarded)
        return self.native_em(x, p)

    def replace(self, p):
        assert np.array_equal(p, self.x)
        assert p.shape == (3,) and np.isfinite(p).all() and (p > 0).all()
        self.rb_calls += 1
        old = self.rb.hist[-self.rb.W:]
        reference = self.native_rb(p)
        if self.arm == "identity" or not old:
            self.forwarded = reference
            return reference
        robust, scalar, clipped, err = corrections(np.asarray(old), self.rb.k)
        direction = float(np.max(np.abs(robust-scalar)))
        self.max_scalar_delta = max(self.max_scalar_delta, direction)
        self.changed += int(direction > .01)
        self.clip_rows += clipped
        self.exposure_error = max(self.exposure_error, err)
        strength = self.rb.alpha
        a = p * np.exp(-strength*robust)
        b = p * np.exp(-strength*scalar)
        a /= a.sum()
        b /= b.sum()
        candidate = a if self.arm == "robust" else b
        assert np.isfinite(candidate).all() and (candidate > 0).all()
        self.max_posterior_move = max(self.max_posterior_move, float(np.max(np.abs(candidate-reference))))
        self.max_scalar_posterior_move = max(self.max_scalar_posterior_move, float(np.max(np.abs(a-b))))
        self.forwarded = candidate
        return candidate

    def receipt(self):
        return dict(calls=self.calls, rb_calls=self.rb_calls, em_calls=self.em_calls,
                    above_001=self.changed, clipped_history_occurrences=self.clip_rows,
                    exposure_max_error=self.exposure_error, max_direction_delta=self.max_scalar_delta,
                    max_stock_posterior_delta=self.max_posterior_move,
                    max_scalar_posterior_delta=self.max_scalar_posterior_move)
