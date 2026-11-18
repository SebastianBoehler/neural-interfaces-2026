"""Change cognitive odds at the native raw FBCNet family, including after refits."""

import copy

import numpy as np
import torch


IDS = tuple(range(47, 53))
ATOL = 2e-15


def project(before, subtraction_share):
    result = before.copy()
    cognitive = before[..., 1] + before[..., 2]
    result[..., 1] = cognitive * subtraction_share
    result[..., 2] = cognitive - result[..., 1]
    assert np.array_equal(result[..., 0], before[..., 0])
    assert np.max(np.abs(result.sum(axis=-1) - before.sum(axis=-1))) <= ATOL
    return result


class RawOdds:
    def __init__(self, model, actor):
        self.base = model.nets
        self.model = model
        self.native_mix = model.mix6
        self.w = model.w[1 + np.asarray(IDS)]
        assert np.allclose(self.w, 1/90, atol=1e-16, rtol=0)
        assert model.ft.member["fbraw"] == IDS[0] and tuple(model.ft.slots["fbraw"]) == IDS
        group = [stream for stream, members in self.base.groups
                 if any(i in IDS for i, _ in members)]
        assert len(group) == 1 and not group[0].align
        self.actor = copy.deepcopy(actor) if actor is not None else None
        self.reset_audit()

    def reset_audit(self):
        self.calls = self.moved = self.late = self.shadow_calls = 0
        self.max_pre = self.max_late = 0.
        self.actor_q = self.expected = self.last_mix = self.shadow_mix = self.routed_mix = None
        if self.actor is not None:
            with torch.no_grad(), torch.jit.optimized_execution(False):
                assert self.actor.reset().item() == 0

    @property
    def last_inputs(self):
        return self.base.last_inputs

    def reset_state(self):
        self.base.reset_state()
        self.reset_audit()

    def __call__(self, x):
        assert x.shape == (41, 480) and x.dtype == np.float32
        members = self.base(x)
        self.indices = [j for j, (i, _) in enumerate(members) if i in IDS]
        assert tuple(members[j][0] for j in self.indices) == IDS
        self.before = np.stack([members[j][1] for j in self.indices])
        self.calls += 1
        self.last_mix = self.shadow_mix = self.routed_mix = None
        self.expected = self.before
        if self.actor is None:
            return members
        inp = torch.from_numpy(np.ascontiguousarray(x)[None]).to(self.base.dev)
        with torch.no_grad(), torch.jit.optimized_execution(False):
            p = self.actor(inp)
        assert p.shape == (1, 3) and p.dtype == torch.float64
        self.actor_q = p.cpu().numpy()[0]
        assert np.isfinite(self.actor_q).all() and (self.actor_q >= 0).all()
        assert abs(float(self.actor_q.sum()) - 1) <= 1e-12
        cognitive = self.actor_q[1:].sum()
        assert cognitive > 0
        self.share = self.actor_q[1] / cognitive
        self.expected = project(self.before, self.share)
        self.base_delta = ((self.expected - self.before) * self.w[:, None]).sum(axis=0)
        assert self.base_delta[0] == 0 and abs(self.base_delta.sum()) <= ATOL
        delta = float(np.abs(self.expected - self.before).max())
        self.max_pre = max(self.max_pre, delta)
        self.moved += int(delta > .01)
        for j, p in zip(self.indices, self.expected, strict=True):
            members[j] = (members[j][0], p)
        return members

    def mix6(self, mix, members, p_ft):
        if self.actor is None or "fbraw" not in p_ft:
            out = self.native_mix(mix, members, p_ft)
        else:
            # Obtain the original raw family term under this arm's actual live refit.
            frozen = dict(members)
            frozen.update(zip(IDS, self.before, strict=True))
            family = (self.w[:, None] * self.before).sum(axis=0)
            amount = self.w.sum()
            adjusted = np.clip(family / amount + p_ft["fbraw"] - self.before[0], 1e-6, None)
            native = amount * adjusted / adjusted.sum()
            changed = project(native, self.share)
            delta = (1 - self.model.w_rest) * (changed - native)
            unprojected = self.native_mix(mix - self.base_delta, frozen, p_ft)
            out = unprojected + delta
            assert out[0] == unprojected[0] and abs(out.sum() - unprojected.sum()) <= ATOL
            assert np.isfinite(out).all() and (out >= 0).all()
            self.max_late = max(self.max_late, float(np.abs(delta).max()))
            self.late += 1
        self.last_mix = out.copy()
        return out

    def check(self, trace, row):
        v6 = trace["v6"]
        actual = np.stack([v6["members"][i] for i in IDS])
        assert np.array_equal(actual, self.expected), "NetBank did not consume the expected raw posteriors"
        assert np.array_equal(trace["mix"], self.shadow_mix), "shadow did not consume the changed base mix"
        assert np.array_equal(v6["mix6"], self.last_mix) and np.array_equal(self.routed_mix, self.last_mix), "the routed tail did not consume mix6"
        assert self.calls == self.shadow_calls == row + 1
        return actual

    def receipt(self):
        return dict(calls=self.calls, above_001=self.moved, warm_calls=self.late,
                    max_pre=self.max_pre, max_warm=self.max_late)


def install(model, actor):
    odds = RawOdds(model, actor)
    native_shadow = model.shadow.step
    native_route = model.routeB.stepx

    def shadow(x, u, mix):
        odds.shadow_mix = mix.copy()
        odds.shadow_calls += 1
        return native_shadow(x, u, mix)

    def route(x, u, mix, *args):
        odds.routed_mix = mix.copy()
        return native_route(x, u, mix, *args)

    model.nets = odds
    model.mix6 = odds.mix6
    model.shadow.step = shadow
    model.routeB.stepx = route
    return odds
