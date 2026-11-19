"""Prove the true gated personal-core insertion at RouteB's actual Tail."""

import numpy as np


class RouteProbe:
    def __init__(self, model):
        self.route = model.routeB
        self.original = self.route.stepx
        self.tail = self.route.tail.step
        self.gates = []
        self.errors = []
        self.last = None
        self.route.stepx = self.stepx
        self.route.tail.step = self.consume

    def stepx(self, x, u, mix, core, extra, contrast, identity):
        self.last = mix
        j, ramp = -1, 0.
        if contrast is not None:
            j, ramp = int(contrast[0]), float(contrast[2])
        if identity is not None and identity[2] > ramp:
            j, ramp = int(identity[0]), float(identity[2])
        self.gates.append((j, ramp))
        p, route = mix, self.route
        if ramp > 0:
            values = np.asarray(route.U + [u])
            features = u - values.mean(0)
            if len(route.U) >= 4:
                features = features / np.maximum(values.std(0), 1e-3)
            logits = features[None] @ route.W[j].T + route.c[j]
            personal = np.exp(logits[0] - logits.max())
            personal /= personal.sum()
            amount = ramp * route.a
            p = (mix - amount * core + route.g * amount * personal) / (1 - amount + route.g * amount)
        self.expected = p
        return self.original(x,u,mix,core,extra,contrast,identity)

    def consume(self, x, u, mix, *args):
        error = float(np.abs(mix - self.expected).max())
        assert error <= 2e-14
        self.errors.append(error)
        return self.tail(x,u,mix,*args)

    def summary(self):
        assert len(self.errors) == len(self.gates)
        return dict(gates=self.gates, routed=sum(r > 0 for _,r in self.gates), max_tail_mix_error=max(self.errors))
