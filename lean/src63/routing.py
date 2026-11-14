"""Routing of a recognised returning person: personal head into the mix, a tail of its own, own-space stage."""
import sys

import numpy as np

from .gates import ContrastGate, IdentityGate
from .numerics import softmax
from .personal import OwnStage
from .stages import Tail


class Route:
    """Per window: the two gates name a public person j with ramp r (the larger ramp wins; they read the views and
    the unrouted posteriors only). For r > 0, core1's share ``a`` of the mix is partly replaced by j's personal
    head h (logistic regression on the 256-PC view standardised over the session so far) at ``g`` times the weight:
    mix' = (mix - r a core1 + g r a h) / (1 - r a + g r a). The routed tail runs on mix' (on the plain mix until
    someone is recognised); the own-space stage then acts on routed windows."""

    def __init__(self, z, a):
        self.W, self.c = (np.asarray(z[k], dtype=np.float64) for k in ("route_ps1_W", "route_ps1_c"))
        self.g, self.a, self.n_tsem = float(z["route_slot_g"]), float(a), z["tsem_V"].shape[1]
        self.contrast, self.identity, self.own, self.tail = ContrastGate(z), IdentityGate(z), OwnStage(z), Tail(z)
        self.observes = True  # v6.3: False for a Route that shares another Route's stage, which observes
        self.reset_state()

    def reset_state(self):
        self.report()
        self.U, self.Q, self.last = [], [], {}
        for part in (self.tail, self.identity, self.own):
            part.reset_state()

    def report(self):
        """One stderr line per session on what the own-space stage did, so that a stage that silently never ran
        shows in the evaluation log."""
        log = getattr(self, "log", None)
        if log and log["n"]:
            print(f"[own-stage] windows {log['n']} routed {log['routed']} degenerate {log['degen']} "
                  f"staged {log['staged']} errors {log['err']}{' first: ' + log['first'] if log['first'] else ''}",
                  file=sys.stderr, flush=True)
        self.log = dict(n=0, routed=0, degen=0, staged=0, err=0, first="")

    def __del__(self):  # the evaluator drops a session's copy without a call: report then
        try:
            self.report()
        except Exception:
            pass

    def step(self, x, u, mix, core1, shadow):
        """x: window; u: 256-PC view; mix, core1: member mix and core1 posterior; shadow: unrouted posterior."""
        t, log = len(self.U), self.log
        j, r = -1, 0.0
        contrast, identity = self.contrast.step(self.U, self.Q), self.identity.step(x)
        if contrast is not None:
            j, r = contrast[0], contrast[2]
        if identity is not None and identity[2] > r:
            j, r = identity[0], identity[2]
        p = mix
        if r > 0:
            A = np.asarray(self.U + [u])
            m = A.mean(0)
            v = (u - m) / np.maximum(A.std(0), 1e-3) if t >= 4 else u - m
            h = softmax(v[None] @ self.W[j].T + self.c[j])[0]
            o = (mix - r * self.a * core1 + self.g * r * self.a * h) / (1 - r * self.a + self.g * r * self.a)
            if np.isfinite(o).all():
                p = o
        out, stage = self.tail.step(x, u[:self.n_tsem], p), None
        log["n"] += 1
        log["routed"] += int(r > 0)
        try:  # the stage is optional: any failure leaves the window's output as it is
            with np.errstate(all="ignore"):
                if self.observes:
                    log["degen"] += int(not self.own.observe(x))
                else:  # v6.3 RouteB: the shared stage observed this window in v5.1b's Route already
                    log["degen"] += int(self.own.cur is None)
                staged = self.own.apply(out, j, r) if r > 0 and self.own.cur is not None else None
                if staged is not None:
                    out, stage = staged[0], (j, r, *staged[1])
                    log["staged"] += 1
        except Exception as e:
            log["err"] += 1
            log["first"] = log["first"] or repr(e)[:200]
        self.U.append(u)
        self.Q.append(shadow)
        self.last = dict(contrast=contrast, identity=identity, stage=stage)
        return out
