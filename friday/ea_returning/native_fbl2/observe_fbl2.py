"""Witness the shared FBLight hook at the actual recognized RouteB tail."""

import numpy as np

from fbl2 import BOOST_SHARE, CAP_SHARE, MEMBERS, FBL2Hook


EPS = 2e-14


def personalized(mix, core, head, ramp, gain, share):
    mass = ramp * share
    return (mix - mass * core + gain * mass * head) / (1 - mass + gain * mass)


class ObserveFBLight:
    def __init__(self, model, arm):
        self.hook = FBL2Hook(model, arm)
        self.route = model.routeB
        self.expected = None
        self.consumed = None
        self.positive = 0
        self.share_effect = 0.
        self.warm_effect = 0.
        self.records = []
        self.share = BOOST_SHARE if arm == "fbl2" else CAP_SHARE
        self.route_call, self.tail_call = self.route.stepx, self.route.tail.step
        self.route.stepx, self.route.tail.step = self.stepx, self.tail

    def stepx(self, x, u, mix, core, extra, contrast, identity):
        if self.expected is not None or self.consumed is not None:
            raise ValueError("unexpected native RouteB invocation")
        self.consumed = mix
        j, _, ramp = max((g for g in (contrast, identity) if g is not None),
                         key=lambda g: g[2], default=(-1, 0., 0.))
        routed = mix
        if ramp > 0:
            self.positive += 1
            context = np.asarray(self.route.U + [u])
            v = u - context.mean(0)
            if len(self.route.U) >= 4:
                v = v / np.maximum(context.std(0), 1e-3)
            logits = (v[None] @ self.route.W[j].T + self.route.c[j])[0]
            e = np.exp(logits - logits.max())
            head = e / e.sum()
            routed = personalized(mix, core, head, ramp, self.route.g, self.share)
            old_share = personalized(mix, core, head, ramp, self.route.g, CAP_SHARE)
            self.share_effect = max(self.share_effect, float(np.abs(routed - old_share).max()))
        self.expected = routed
        val = self.route_call(x, u, mix, core, extra, contrast, identity)
        if self.expected is not None:
            raise ValueError("the genuine routed tail did not consume the FBLight mixture")
        return val

    def tail(self, x, u, p, extra=None):
        np.testing.assert_allclose(p, self.expected, rtol=0, atol=EPS)
        self.expected = None
        if extra is None:
            return self.tail_call(x, u, p)
        return self.tail_call(x, u, p, extra)

    def inspect(self, trace, t):
        v = self.hook.model.trace["v6"]
        np.testing.assert_array_equal(v["mix6"], trace["v6"]["mix6"])
        self.hook.check(v)
        n = self.hook
        if self.consumed is not v["mix6"] or n.calls != t + 1:
            raise ValueError("native mix and RouteB did not share the actual posterior")
        self.consumed = None
        base = np.stack([v["members"][i] for i in MEMBERS])
        q = base.mean(0)
        adapted = q
        if n.last_warm:
            adapted = np.clip(q + v["p_ft"]["fbl"] - base[0], 1e-6, None)
            adapted /= adapted.sum()
        np.testing.assert_allclose(n.last_fbl, adapted, rtol=0, atol=EPS)
        expected = n.last_stock
        if self.share == BOOST_SHARE:
            expected = (16 * n.last_stock + adapted) / 17
            if n.last_warm:
                frozen_only = (16 * n.last_stock + q) / 17
                self.warm_effect = max(self.warm_effect, float(np.abs(expected - frozen_only).max()))
        np.testing.assert_allclose(v["mix6"], expected, rtol=0, atol=EPS)
        self.records.append(np.stack((n.last_stock, n.last_fbl, v["extra"], v["rest"], v["label"])))

    def summary(self):
        return dict(self.hook.summary(), positive_route_rows=self.positive,
                    core_share_max_delta=self.share_effect, adapted_mix_max_delta=self.warm_effect)
