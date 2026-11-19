"""Use the authentic C20e outer adapter on the archived returning CPU worker."""

from pathlib import Path

import numpy as np
import torch

from adapter import Adapter
from eye_scope import BASE, pin
from pupil_inputs import TOL


UPSTREAM_ADAPTER = BASE / "ni26_friday_c20e_native_v4_20261119/adapter.py"
ADAPTER_SHA = "278462cc967095ecd9ecca0f03f066716c3717b18858c5c121271e458c609974"


class ReturningAdapter(Adapter):
    def __init__(self, model, arm, templates, x, oracle):
        if Path(Adapter.mix6.__code__.co_filename) != UPSTREAM_ADAPTER:
            raise ValueError("not the authenticated native C20e outer consumer")
        pin(UPSTREAM_ADAPTER, ADAPTER_SHA)
        super().__init__(model, arm, templates)
        self.x, self.oracle = x, oracle
        self.peak = 0.

    @torch.no_grad()
    def pupil_q(self, raw):
        np.testing.assert_array_equal(raw, self.x[self.calls])
        x = torch.from_numpy(np.ascontiguousarray(raw[None]))
        with torch.jit.optimized_execution(False):
            outputs = [m(x) for m in self.actors]
        if len(outputs) != 2 or any(a.shape != (1, 3) or a.dtype != torch.float64 for a in outputs):
            raise ValueError("wrong real frozen pupil output")
        pair = np.stack([a.numpy()[0] for a in outputs])
        target = self.oracle[self.calls]
        np.testing.assert_array_equal(pair.argmax(1), target.argmax(1))
        error = float(np.abs(pair - target).max())
        if error > TOL or not np.isfinite(pair).all() or (pair < 0).any():
            raise ValueError("real raw actor differs from its independent oracle")
        np.testing.assert_allclose(pair.sum(1), 1, rtol=0, atol=1e-12)
        self.peak = max(self.peak, error)
        self.pupil.append(pair)
        return pair.mean(0)
