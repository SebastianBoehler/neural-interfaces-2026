"""Replace RouteB's head evidence at its original weight and consuming tail."""

import numpy as np

from geometry import forward_feature, live_waveform, unit


class RecognizedHead:
    def __init__(self, model, arm, fit, names, who):
        self.arm, self.fit, self.names, self.who = arm, fit, names, who
        self.route = model.routeB
        route = self.route
        assert np.isclose(route.a, 1/16, atol=1e-16, rtol=0)
        assert (route.g, route.kappa, route.n_tsem) == (3., 20., 128)
        assert type(route).__name__ == "RouteB"
        _, self.member = live_waveform(model)
        self.native_route, self.native_tail = route.stepx, route.tail.step
        self.softmax = type(route).step.__globals__["softmax"]
        self.n = self.consumed = self.active = 0
        self.head_max = self.mix_error = self.local_effect = 0.
        self.expected = self.feature = None
        self.ramps, self.inputs = [], []
        self.member.proba = self.tap
        route.stepx, route.tail.step = self.step, self.tail

    def tap(self, x, cache):
        z, p = forward_feature(self.member, x, cache)
        self.feature = z.detach().numpy()[0].copy()
        assert self.feature.shape == self.fit["mean"].shape
        self.n += 1
        return p

    def step(self, x, u, mix, core, extra, contrast, identity):
        route = self.route
        age = len(route.U)
        assert self.n == age + 1 and self.consumed == age
        assert self.feature is not None
        g = contrast
        if identity is not None and identity[2] > (g[2] if g is not None else 0.):
            g = identity
        ramp = 0. if g is None else float(g[2])
        assert 0 <= ramp <= 1
        updated = mix
        self.expected = mix.copy()
        if ramp:
            j = g[0]
            assert str(self.names[j]).split("/")[-1] == self.who
            all_u = np.asarray(route.U + [u])
            center = all_u.mean(0)
            v = (u-center)/np.maximum(all_u.std(0), 1e-3) if age >= 4 else u-center
            native = self.softmax(v[None]@route.W[j].T+route.c[j])[0]
            if self.arm == "placebo":
                use = native
            else:
                z = unit((self.feature-self.fit["mean"])/self.fit["scale"])
                use = self.softmax((8*(self.fit[self.arm]@z))[None])[0]
                assert np.isfinite(use).all() and np.isclose(use.sum(), 1., rtol=0, atol=1e-12)
                updated = mix + route.g*ramp*route.a*(use-native)
            dose = ramp*route.a
            self.expected = (mix-dose*core+route.g*dose*use)/(1-dose+route.g*dose)
            ref = (mix-dose*core+route.g*dose*native)/(1-dose+route.g*dose)
            self.active += 1
            self.head_max = max(self.head_max, float(np.abs(use-native).max()))
            self.local_effect = max(self.local_effect, float(np.abs(self.expected-ref).max()))
        self.ramps.append(ramp)
        result = self.native_route(x, u, updated, core, extra, contrast, identity)
        assert self.consumed == self.n
        return result

    def tail(self, x, u, mix):
        assert mix.shape == (3,) and np.isfinite(mix).all() and (mix > 0).all()
        error = float(np.abs(mix-self.expected).max())
        assert error <= 1e-14, error
        if self.ramps[-1] == 0.:
            np.testing.assert_array_equal(mix, self.expected)
        self.mix_error = max(self.mix_error, error)
        self.inputs.append(mix.copy())
        self.consumed += 1
        return self.native_tail(x, u, mix)

    def receipt(self):
        assert self.n == self.consumed == len(self.ramps)
        return dict(calls=self.n, positive=self.active, routed=self.route.log["routed"],
                    own_staged=self.route.log["staged"], max_head_delta=self.head_max,
                    max_local_mixture_delta=self.local_effect, max_mixture_error=self.mix_error)
