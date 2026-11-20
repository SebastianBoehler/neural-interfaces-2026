"""Observe the live native EA member input and strict-prefix FineTune consumer."""

import numpy as np

from inputs import GROUPS


class ReplaceEA:
    def __init__(self, actor, values):
        self.actor = actor
        self.rows = values
        self.n = 0
        self.ids = {i: slot for slot, ids in GROUPS.items() for i in ids}

    def __getattr__(self, key):
        return getattr(self.actor, key)

    def __call__(self, x):
        members = self.actor(x)
        self.before = dict(members)
        if not self.ids.keys() <= self.before.keys():
            raise ValueError("requested EA member is inactive")
        out = members if self.rows is None else [
            (i, self.rows[self.ids[i]][self.n] if i in self.ids else p) for i, p in members]
        self.n += 1
        self.after = dict(out)
        return out


class ObserveRefit:
    def __init__(self, model):
        self.model = model
        self.history = {slot: [] for slot in model.ft.member}
        self.targets, self.optimizer, self.warm = {}, {}, {}
        self.age = -1
        model.relabel_prefix, self.native_labels = self.labels, model.relabel_prefix
        model.ft._refit, self.native_fit = self.fit, model.ft._refit

    def labels(self, t):
        model = self.model
        p = self.native_labels(t)
        if p is None or t != self.age or model.recB or t in self.targets:
            raise ValueError("unstable native relabel age or recognition latch")
        tail = model.tail6r if model.v6r else model.route.tail
        if len(tail.em.A) != t + 1 or len(tail.tsem.U) != t + 1:
            raise ValueError("the source label chain did not advance the actual age")
        expected = tail.tsem.reapply(t, tail.em.reapply(t))
        if p.shape != (t, 3) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-12):
            raise ValueError("the actor did not provide a normalized strict prefix")
        np.testing.assert_array_equal(p, expected)
        self.targets[t] = p.copy()
        return p

    def fit(self, t, labels, weights):
        ft = self.model.ft
        target = self.targets[t]
        if t in self.optimizer or set(ft.nets) != set(self.history):
            raise ValueError("missing live FineTune family")
        np.testing.assert_array_equal(labels, target.argmax(1))
        np.testing.assert_array_equal(weights, target.max(1))
        for slot, hist in self.history.items():
            if len(hist) != t or len(ft.X[slot]) != t + 1:
                raise ValueError("strict-prior prefix carries current or omitted inputs")
            np.testing.assert_array_equal(np.stack(ft.X[slot][:t]), np.stack(hist))
        before = ft.steps
        ret = self.native_fit(t, labels, weights)
        steps = ft.steps - before
        if steps != getattr(ft, "STEPS_PER_FAMILY", 5) * len(self.history) or ft.dtheta <= 0:
            raise ValueError("real matching four-family optimization was inactive")
        self.optimizer[t] = dict(steps=steps, dtheta=ft.dtheta)
        return ret

    def emission(self, trace):
        for slot, i in self.model.ft.member.items():
            self.history[slot].append(self.model.nets.last_inputs[i].copy())
        if v := trace["v6"]["p_ft"]:
            if self.model.recB or set(v) != set(self.history) or not self.optimizer:
                raise ValueError("warm family emitted after recognition or before refit")
            if any(p.shape != (3,) or not np.isfinite(p).all() or not np.isclose(p.sum(), 1) for p in v.values()):
                raise ValueError("invalid live warm family posterior")
            self.warm[self.age] = {slot: p.copy() for slot, p in v.items()}
