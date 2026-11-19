"""Exact warm-family reweighting at the cap250 final mix and its RouteB share."""

import numpy as np


MEMBERS = tuple(range(59, 65))
CAP_SHARE = 1 / 16
BOOST_SHARE = 1 / 17


class FBL2Hook:
    """Bind one fresh Solver; stock observes, identity exercises, fbl2 doubles."""

    def __init__(self, model, arm):
        if arm not in {"stock", "identity", "fbl2"}:
            raise ValueError("expected a fixed FBL2 arm")
        self.model, self.arm = model, arm
        assert model.routed == 1 and model.r15 and not model.v6r
        assert model.ft.member["fbl"] == MEMBERS[0] and tuple(model.ft.slots["fbl"]) == MEMBERS
        weights = model.w[1 + np.asarray(MEMBERS)]
        assert np.allclose(weights, 1 / 90, atol=1e-16, rtol=0)
        assert model.w_rest == CAP_SHARE and abs(model.routeB.a - CAP_SHARE) <= 1e-16
        self.weight = weights
        self.mass = float(weights.sum())
        self.post_rest = (1 - model.w_rest) * self.mass
        assert abs(self.post_rest - CAP_SHARE) <= 1e-16
        self.stock_inner_share = model.route.a
        self.original_share = model.routeB.a
        self.route_share = BOOST_SHARE if arm == "fbl2" else self.original_share
        self.original_mix = model.mix6
        self.calls = self.warm = self.frozen = self.moved = 0
        self.warm_posterior_delta = self.mix_delta = 0.
        self.last_warm = None
        self.last_fbl = self.last_stock = self.last_mix = None
        if arm != "stock":
            model.routeB.a = self.route_share
            model.mix6 = self.mix6

    def project(self, native, members, warm):
        """Accept a current native outer mix and the untouched owning inputs."""
        assert members and all(i in members for i in MEMBERS)
        assert native.shape == (3,) and np.isfinite(native).all()
        original = np.stack([members[i] for i in MEMBERS])
        family = (self.weight[:, None] * original).sum(0) / self.mass
        changed = family
        if "fbl" in warm:
            values = np.clip(family + warm["fbl"] - original[0], 1e-6, None)
            changed = values / values.sum()
            self.warm += 1
            self.warm_posterior_delta = max(self.warm_posterior_delta, float(np.abs(changed-family).max()))
        else:
            self.frozen += 1
        assert np.isfinite(changed).all() and (changed >= 0).all()
        assert abs(float(changed.sum()) - 1) <= 1e-12
        out = native
        if self.arm == "fbl2":
            out = (native + self.post_rest * changed) / (1 + self.post_rest)
            movement = float(np.abs(out-native).max())
            self.mix_delta = max(self.mix_delta, movement)
            self.moved += int(movement > 0)
            assert abs(float(out.sum()) - 1) <= 1e-12
        self.calls += 1
        self.last_stock, self.last_mix, self.last_fbl = native, out, changed
        self.last_warm = "fbl" in warm
        return out

    def mix6(self, mix, members, warm):
        value = self.original_mix(mix, members, warm)
        if not members:
            return value
        return self.project(value, members, warm)

    def check(self, trace):
        """Verify the caller's actual consumed outer mix after one Solver window."""
        assert self.model.route.a == self.stock_inner_share
        assert self.model.routeB.a == self.route_share
        if self.arm != "stock":
            assert trace["mix6"] is self.last_mix
            assert self.last_warm == ("fbl" in trace["p_ft"])
        else:
            self.project(trace["mix6"], trace["members"], trace["p_ft"])
        assert self.calls == self.warm + self.frozen

    def summary(self):
        return dict(arm=self.arm,calls=self.calls,warm_rows=self.warm,frozen_rows=self.frozen,
                    original_share=self.original_share,route_share=self.route_share,
                    fbl_warm_delta=self.warm_posterior_delta,max_mix_delta=self.mix_delta,moved=self.moved)
