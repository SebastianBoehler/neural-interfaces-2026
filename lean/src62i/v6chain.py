"""v6: a second chain for sessions the gate has not recognised; recognised sessions are v5.1b from their first routed
window on (docs/V6_DESIGN.md, ledger §152, §120)."""
import sys

import numpy as np

from .ensemble import Ensemble, submit
from .extras import EyeNet, RestMember
from .finetune import FineTune
from .routing import Route
from .stages import SessionEM, Tail


class SessionEMx(SessionEM):
    """SessionEM whose view carries extra columns per window (the eye net's three log posteriors)."""

    extra = None

    def features(self, x):
        return np.concatenate([super().features(x), self.extra])


class TailX(Tail):
    def __init__(self, z):
        super().__init__(z)
        self.em = SessionEMx(z)

    def step(self, x, u, mix, extra=None):
        if extra is not None:
            self.em.extra = np.asarray(extra, dtype=np.float64)
        return super().step(x, u, mix)


class GivenGate:
    """A gate whose decision of the window is given (v5.1b's own gate on its own shadow): the gate-unchanged v6.2."""

    def __init__(self):
        self.next = None

    def reset_state(self):
        self.next = None

    def step(self, *args):
        return self.next


class RouteB(Route):
    """v6.2: v5.1b's routing step (personal head into the mix, routed tail, own-space stage) on the new mix, with the
    eye columns in its tail's view, and with v5.1b's own gate decisions of the window (``contrast``, ``identity``)."""

    def __init__(self, z, a):
        super().__init__(z, a)
        self.tail, self.contrast, self.identity = TailX(z), GivenGate(), GivenGate()
        self.reset_state()

    def stepx(self, x, u, mix, core1, extra, contrast, identity):
        self.tail.em.extra = np.asarray(extra, dtype=np.float64)
        self.contrast.next, self.identity.next = contrast, identity
        return self.step(x, u, mix, core1, None)


class Ensemble6(Ensemble):
    """v5.1b runs unchanged on every window. While no window of the session has been routed, a second chain runs
    beside it and gives the output: the mix with each fine-tuned family's change (``FineTune``) and the rest member
    (weight ``v6_rest_w`` of the total), then rebalancing -> SessionEM (view + eye columns) -> TSEM. At the first
    window either gate routes, the v6 chain stops for the session and the output is v5.1b's routed tail. The gates
    never see the v6 chain."""

    def __init__(self, z, dev, root):
        super().__init__(z, dev, root)
        self.eye, self.rest, self.ft, self.tail6 = EyeNet(z, dev, root), RestMember(z, dev, root), FineTune(z, self.w, dev, root), TailX(z)
        self.w_rest = float(z["v6_rest_w"])
        # v6.1: fine-tuning labels from the v6 chain without fine-tuning (mix + rest, eye view; a third tail); 0: v5.1b's output
        self.v6r = bool(int(z["v6_ft_labels"]))
        self.tail6r = TailX(z) if self.v6r else None
        # v6.2 (v6_routed, gate-unchanged form of ledger s201): 0 = v6.1; 1 = v6.2-safe, 2 = v6.2. v5.1b runs unchanged and
        # its gates decide recognition (the shadow they read is v5.1b's). The output is RouteB: v5.1b's routing step on
        # mix_B = (1 - w_rest)(mix + FT change) + w_rest rest with the eye view in its tail, the personal head at core1's
        # share w0 (1 - w_rest), and v5.1b's gate decisions of the window. Before recognition (r = 0) RouteB's tail is
        # v6.1's second chain. Fine-tuning: v6.1's labels before recognition; at recognition every stored label is
        # replaced by v5.1b's routed output, which labels the windows after it; v6.2-safe stops fine-tuning there.
        self.routed = int(z["v6_routed"]) if "v6_routed" in z else 0
        if self.routed:
            self.routeB = RouteB(z, self.w[0] * (1 - float(z["v6_rest_w"])))
        self.reset_state()

    def reset_state(self):
        super().reset_state()
        self.report6()
        for part in (self.eye, self.rest, self.ft, self.tail6, self.tail6r, getattr(self, "routeB", None)):
            if part is not None:
                part.reset_state()
        self.active, self.recB, self.v5hist, self.dead = True, False, [], False

    def report6(self):
        """One stderr line per session: how long the v6 chain ran and what fine-tuning did, so that a part that
        silently never ran shows in the evaluation log."""
        log = getattr(self, "log6", None)
        if log and log["n"]:
            print(f"[v6] windows {log['n']} v6 {log['v6']} recognised_at {log['rec']} refits {self.ft.refits} "
                  f"steps {self.ft.steps} dtheta {self.ft.dtheta:.3g} swapped {log['ft']} capped {self.ft.capped} "
                  f"over {self.ft.over} ft_s {self.ft.ft_sec:.0f} errors {log['err']}"
                  f"{' first: ' + log['first'] if log['first'] else ''}", file=sys.stderr, flush=True)
        self.log6 = dict(n=0, v6=0, rec=-1, ft=0, err=0, first="")

    def __del__(self):
        try:
            self.report6()
        except Exception:
            pass

    def mix6(self, mix, members, p_ft):
        """mix with every fine-tuned family's posterior moved by (fine-tuned - frozen) of its tuned member (clipped at
        1e-6, renormalised; the dev form of pkg.py), then the rest member at weight w_rest of the total."""
        m = mix.copy()
        for slot, p in p_ft.items():
            ids, k = self.ft.slots[slot], self.ft.member[slot]
            w = self.w[1 + ids]
            fam = (w[:, None] * np.stack([members[i] for i in ids])).sum(0)
            F = np.clip(fam / w.sum() + p - members[k], 1e-6, None)
            m = m - fam + w.sum() * F / F.sum()
        return (1 - self.w_rest) * m + self.w_rest * self.rest_p

    def step(self, x):
        return self.step62(x) if self.routed else self.step61(x)

    def step62(self, x):
        """v6.2 (gate-unchanged): v5.1b decides recognition; the output is RouteB on the new mix with the eye view."""
        cores = submit(self._tangents, x)
        nets = self.nets(x)
        core1, u, core2 = cores.result()
        mix = self.w[0] * core1
        for i, p in nets:
            mix = mix + self.w[1 + i] * p
        mix = mix + self.w[-1] * core2
        if not np.isfinite(mix).all():
            mix = np.full(len(mix), 1.0 / len(mix))
        n_tsem, log = self.route.n_tsem, self.log6
        shadow = self.shadow.step(x, u[:n_tsem], mix)
        out_v5 = self.route.step(x, u, mix, core1, shadow)  # v5.1b, unchanged: its gates decide recognition
        self.v5hist.append(out_v5)
        log["n"] += 1
        gc, gi = self.route.last["contrast"], self.route.last["identity"]
        new = not self.recB and max([g[2] for g in (gc, gi) if g is not None], default=0.0) > 0
        if new:
            self.recB, log["rec"] = True, log["n"] - 1
        if self.dead:  # an earlier window of this session raised: v5.1b's output for the rest of the session
            return self._fallback(core1, mix, u, shadow, out_v5, gc, gi)
        try:  # v6.2h: any error in the additions keeps v5.1b's output (as v6.1's chain does), never the run
            if new:
                self.ft.relabel(self.v5hist[:-1])  # every earlier window now labelled by v5.1b's output
            extra, self.rest_p = self.eye.step(x), self.rest.step(x)
            if not self.recB and self.v6r:  # unrecognised: the v6 chain without fine-tuning labels this window (v6.1)
                label = self.tail6r.step(x, u[:n_tsem], self.mix6(mix, {}, {}), extra)
            else:
                label = out_v5
            p_ft = self.ft.step(dict(nets), self.nets.last_inputs, label) if (self.routed == 2 or not self.recB) else {}
            mB = self.mix6(mix, dict(nets), p_ft)
            if not np.isfinite(mB).all():
                raise FloatingPointError("v6.2 mix not finite")
            out = self.routeB.stepx(x, u, mB, core1, extra, gc, gi)
            if not np.isfinite(out).all():
                raise FloatingPointError("v6.2 output not finite")
        except Exception as e:  # counted (log err, first error in the [v6] line); the session stays on v5.1b from here
            self.dead, log["err"] = True, log["err"] + 1
            log["first"] = log["first"] or repr(e)[:200]
            print(f"[v6] error at window {log['n'] - 1}: {repr(e)[:200]}; v5.1b's output for the rest of the session",
                  file=sys.stderr, flush=True)
            return self._fallback(core1, mix, u, shadow, out_v5, gc, gi)
        log["v6"] += 1
        log["ft"] += int(bool(p_ft))
        self.trace = dict(core1=core1, mix=mix, u=u, shadow=shadow, out=out, out_v5=out_v5, active=not self.recB,
                          v6=dict(extra=extra, rest=self.rest_p, p_ft=p_ft, mix6=mB, members=dict(nets), label=label,
                                  ft_inputs={i: self.nets.last_inputs[i] for i in self.ft.member.values()}),
                          contrast=gc, identity=gi, stage=self.routeB.last["stage"])
        return out

    def _fallback(self, core1, mix, u, shadow, out_v5, gc, gi):
        self.trace = dict(core1=core1, mix=mix, u=u, shadow=shadow, out=out_v5, out_v5=out_v5, active=False, v6={},
                          contrast=gc, identity=gi, stage=None, dead=True)
        return out_v5

    def step61(self, x):
        cores = submit(self._tangents, x)
        nets = self.nets(x)
        core1, u, core2 = cores.result()
        mix = self.w[0] * core1
        for i, p in nets:
            mix = mix + self.w[1 + i] * p
        mix = mix + self.w[-1] * core2
        if not np.isfinite(mix).all():
            mix = np.full(len(mix), 1.0 / len(mix))
        shadow = self.shadow.step(x, u[:self.route.n_tsem], mix)
        out = self.route.step(x, u, mix, core1, shadow)
        log = self.log6
        log["n"] += 1
        ramps = [g[2] for g in (self.route.last["contrast"], self.route.last["identity"]) if g is not None]
        if self.active and max(ramps, default=0.0) > 0:
            self.active, log["rec"] = False, log["n"] - 1
        final, v6 = out, {}
        if self.active:
            try:
                extra, self.rest_p = self.eye.step(x), self.rest.step(x)
                label = out
                if self.v6r:  # v6.1: the v6 chain without fine-tuning gives this window's label for later refits
                    label = self.tail6r.step(x, u[:self.route.n_tsem], self.mix6(mix, {}, {}), extra)
                p_ft = self.ft.step(dict(nets), self.nets.last_inputs, label)
                m6 = self.mix6(mix, dict(nets), p_ft)
                if not np.isfinite(m6).all():
                    raise FloatingPointError("v6 mix not finite")
                final = self.tail6.step(x, u[:self.route.n_tsem], m6, extra)
                v6 = dict(extra=extra, rest=self.rest_p, p_ft=p_ft, mix6=m6, members=dict(nets), label=label,
                          ft_inputs={i: self.nets.last_inputs[i] for i in self.ft.member.values()})
                log["v6"] += 1
                log["ft"] += int(bool(p_ft))
            except Exception as e:  # counted; the window keeps v5.1b's output, and the chain stops for the session
                self.active, log["err"] = False, log["err"] + 1
                log["first"] = log["first"] or repr(e)[:200]
        self.trace = dict(core1=core1, mix=mix, u=u, shadow=shadow, out=final, out_v5=out, active=self.active, v6=v6,
                          **self.route.last)
        return final
