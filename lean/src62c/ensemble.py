"""The model: member mix, unrouted tail, routing. All state is per session and causal."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .inputs import to_windows
from .nets import NetBank
from .routing import Route
from .stages import Tail
from .tangent import TangentCore

_POOL, _POOL_PID = None, None


def submit(fn, *args):
    """fn(*args) on one process-local worker thread (a forked process cannot reuse its parent's thread)."""
    global _POOL, _POOL_PID
    if _POOL is None or _POOL_PID != os.getpid():
        _POOL, _POOL_PID = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eeg-tangent"), os.getpid()
    return _POOL.submit(fn, *args)


def with_prefix(z, prefix):
    return {k[len(prefix):]: v for k, v in z.items() if k.startswith(prefix)}


class Ensemble:
    """Members and their weights in ``member_weights`` order: core1, the nets, the Hilbert core. ``predict`` takes
    the harness batch (the evaluator sends one window per call) and handles it window by window."""

    def __init__(self, z, dev, root):
        w = z["member_weights"].astype(np.float64)
        self.w = w / w.sum()
        n = int(z["n_nets"])
        self.core1, self.core2 = TangentCore(with_prefix(z, "ts_")), TangentCore(with_prefix(z, "ts2_"))
        self.nets = NetBank([with_prefix(z, f"net{i}_") for i in range(n)], self.w[1:1 + n] != 0, dev, root)
        self.view_m = z["tsem_m"].astype(np.float64)  # 256-PC view of core1's features: TSEM's PCs, then routing's
        self.view_V = [z[k].astype(np.float64) for k in ("tsem_V", "route_V_hi")]
        self.shadow, self.route = Tail(z), Route(z, self.w[0])
        self.perm, self.trace = None, {}  # perm: set by the solver from the runtime montage

    def reset_state(self):
        for part in (self.core1, self.core2, self.nets, self.shadow, self.route):
            part.reset_state()

    def _tangents(self, x):
        p1, z = self.core1.step(x)
        d = z - self.view_m
        return p1, np.r_[d @ self.view_V[0], d @ self.view_V[1]], self.core2.step(x)[0]

    def step(self, x):
        """One window (C, T), float32 -> class posterior. The cores run on the worker thread while the nets run
        here; both finish before the stages."""
        cores = submit(self._tangents, x)
        nets = self.nets(x)
        core1, u, core2 = cores.result()
        mix = self.w[0] * core1
        for i, p in nets:
            mix = mix + self.w[1 + i] * p
        mix = mix + self.w[-1] * core2
        if not np.isfinite(mix).all():  # never let NaN enter session state
            mix = np.full(len(mix), 1.0 / len(mix))
        shadow = self.shadow.step(x, u[:self.route.n_tsem], mix)
        out = self.route.step(x, u, mix, core1, shadow)
        self.trace = dict(core1=core1, mix=mix, u=u, shadow=shadow, out=out, **self.route.last)
        return out

    def predict(self, X):
        labels = [int(self.step(x).argmax()) for x in to_windows(X, self.perm)]
        return torch.as_tensor(labels, dtype=torch.long)
