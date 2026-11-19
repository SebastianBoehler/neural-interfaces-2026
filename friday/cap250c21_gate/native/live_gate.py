"""Run an alternate live native tail and select its current output without feedback."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from fbl2 import FBL2Hook
from mirror import Outer


class LiveSelector:
    retire_on_recognition = False

    def __init__(self, model, audit=False):
        if model.v5hist or model.cp.t or model.dead:
            raise ValueError("live selection requires a fresh native session")
        self.model = model
        self.outer = Outer(model)
        fields = ("routed", "r15", "v6r", "ft", "w", "w_rest", "route", "mix6")
        view = SimpleNamespace(routeB=self.outer.route, **{key: getattr(model, key) for key in fields})
        self.hook = FBL2Hook(view, "fbl2")
        if self.outer.route.tail.em is model.routeB.tail.em or self.outer.cp.P is model.cp.P:
            raise ValueError("alternate native downstream state overlaps stock")
        self.native_predict, self.native_mix = model.predict, model.mix6
        self.native_route, self.native_terminal = model.routeB.stepx, model.cp.step
        model.mix6, model.routeB.stepx, model.cp.step = self.mix, self.route, self.terminal
        self.pending = self.frame = self.stock = None
        self.age = self.eligible = 0
        self.recognized = False
        self.recognition_age = -1
        self.protect = None
        self.probability = None
        self.audit = audit
        self.events, self.stock_stage, self.selected = [], [], []

    def mix(self, mix, members, warm):
        p = self.native_mix(mix, members, warm)
        if self.pending is not None or not members:
            raise ValueError("invalid native stock mixer invocation")
        q = None if self.retire_on_recognition and self.model.recB else self.hook.project(p, members, warm)
        self.pending = (p, q, members, warm)
        return p

    def route(self, x, u, mix, core, extra, contrast, identity):
        p, q, members, warm = self.pending
        if mix is not p or self.frame is not None:
            raise ValueError("native stock RouteB did not consume its current mix")
        self.pending = None
        self.frame = (x, u, p, q, core, extra, contrast, identity, self.model.recB)
        if self.audit:
            self.events.append(deepcopy((x, u, p, core, extra, contrast, identity, self.model.recB, members, warm)))
        return self.native_route(x, u, mix, core, extra, contrast, identity)

    def terminal(self, x, raw, count, recognized):
        if self.frame is None or recognized != self.frame[-1] or self.stock is not None or x is not self.frame[0]:
            raise ValueError("current native terminal is not the current captured route input")
        p = self.native_terminal(x, raw, count, recognized)
        self.stock = p
        if self.audit:
            self.stock_stage.append(np.stack((raw, count, p)))
        return p

    def predict(self, x):
        y = self.native_predict(x)
        event, stock = self.frame, self.stock
        if event is None or stock is None or self.pending is not None or int(y[0]) != int(stock.argmax()):
            raise ValueError("stock must complete its own current decision first")
        newly_seen = any(g is not None and g[2] > 0 for g in event[6:8])
        seen = self.recognized or newly_seen
        if self.model.recB != seen or event[-1] != seen:
            raise ValueError("selector does not match actual native recognition latch")
        if seen and not self.recognized:
            self.recognition_age = self.age
        self.recognized = seen
        if not (self.retire_on_recognition and seen):
            self.outer.step(event, "hard")
        elif event[3] is not None:
            raise ValueError("retired branch was mixed after recognition")
        self.protect = self.age < 20 or seen
        self.probability = stock if self.protect else self.outer.terminal[-1]
        self.eligible += int(not self.protect)
        self.age += 1
        if self.audit:
            self.selected.append(self.probability.copy())
        self.frame = self.stock = None
        if self.model.dead or self.model.cf.dead or self.model.cp.dead or self.model.ft.over or self.model.log6["err"]:
            raise ValueError("native shared upstream failed or fell back")
        return y if self.protect else torch.tensor([int(self.probability.argmax())], dtype=torch.long)

    def summary(self):
        retired = self.retire_on_recognition and self.recognized
        n = self.recognition_age if retired else self.age
        stop = None if retired else self.model.cp.stopped
        if (self.outer.cp.t != n or self.hook.calls != n or len(self.outer.terminal) != n
                or self.outer.cp.stopped != stop or self.model.log6["rec"] != self.recognition_age):
            raise ValueError("stock or alternate native activity differs from the sticky recognition scope")
        return dict(rows=self.age, recognized=self.recognition_age, eligible=self.eligible,
                    alternate_rows=n, retired_at=self.recognition_age if retired else None,
                    alternate_refits=self.outer.cp.refits, alternate_fused=self.outer.cp.fused,
                    alternate_fit_ages=self.outer.fit, hook=self.hook.summary(), native_stop=self.model.cp.stopped)

    def stages(self):
        alternate = np.stack((self.outer.raw, self.outer.count, self.outer.terminal), axis=1).reshape(-1,3,3)
        return np.asarray(self.stock_stage), alternate, np.asarray(self.selected)


class RetiringSelector(LiveSelector):
    """Stop the alternate downstream clock on the actual sticky recognition frame."""

    retire_on_recognition = True
