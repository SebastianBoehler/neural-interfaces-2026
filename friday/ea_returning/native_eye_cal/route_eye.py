"""Blend the calibrated Eye head only under the actual native winning gate."""

import numpy as np

from eye_cal_scope import MASS
from eye_embedding import embeddings
from native_eye import ObserveEye


def winner(gates, names):
    gate = gates["contrast"]
    ident = gates["identity"]
    if ident is not None and (gate is None or ident[2] > gate[2]):
        gate = ident
    return ("", 0.) if gate is None or gate[2] <= 0 else (names[gate[0]], float(gate[2]))


class RouteEye(ObserveEye):
    def __init__(self, model, heads, networks, names):
        super().__init__(model)
        self.heads, self.networks, self.names = heads, networks, names
        self.step = model.eye.step
        model.eye.step = self.readout
        self.raw, self.personal, self.ramp, self.person = [], [], [], []
        self.peak, self.checked = 0., False

    def readout(self, x):
        raw = self.step(x)
        who, r = winner(self.model.route.last, self.names)
        self.raw.append(raw.copy())
        if r <= 0 or who not in self.heads:
            self.personal.append(np.zeros(3))
            self.person.append("")
            self.ramp.append(0.)
            return raw
        center, scale, coef, intercept = self.heads[who]
        h, error = embeddings(self.networks, self.traces[-1][None], check=not self.checked)
        self.peak, self.checked = max(self.peak, error), True
        logits = (h[0] - center) / scale @ coef.T + intercept
        person = np.exp(logits - logits.max())
        person /= person.sum()
        p = np.exp(raw)
        p /= p.sum()
        a = MASS * r
        self.personal.append(person)
        self.person.append(who)
        self.ramp.append(r)
        return np.log(np.clip((1 - a) * p + a * person, 1e-3, 1))

    def inspect(self, trace, t):
        super().inspect(trace, t)
        own = dict(contrast=trace["contrast"], identity=trace["identity"])
        forwarded = dict(contrast=self.model.routeB.contrast.next, identity=self.model.routeB.identity.next)
        if forwarded != own or len(self.raw) != t + 1:
            raise ValueError("Eye head did not use the actual RouteB winning gate")
        expected, ramp = winner(own, self.names)
        if (expected not in self.heads and self.ramp[-1] != 0
                or expected in self.heads and (self.person[-1], self.ramp[-1]) != (expected, ramp)):
            raise ValueError("the calibrated actor used a different gate or ramp")

    def arrays(self):
        return dict(super().arrays(), **{k: np.asarray(getattr(self, k)) for k in
                                        ("raw", "personal", "ramp", "person")})
