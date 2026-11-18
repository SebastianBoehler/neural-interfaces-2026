"""Adjust the live fine-tuned family from past out-of-fit soft-label agreement."""
import numpy as np


class Reliability:
    def __init__(self, mode):
        assert mode in ('adaptive', 'fixed', 'neutral')
        self.mode = mode
        self.past = {}
        self.reads = {}
        self.n_changed = 0

    def read(self, slot):
        if self.mode == 'adaptive':
            gain, n = self.past.get(slot, (0., 0))
            return float(np.clip(.75 + 8 * gain / (20 + n), .25, 1.))
        return .75 if self.mode == 'fixed' else 1.

    def observe(self, slot, frozen, adapted, teacher):
        gain, n = self.past.get(slot, (0., 0))
        improvement = .5 * (np.sum((frozen-teacher)**2) - np.sum((adapted-teacher)**2))
        assert np.isfinite(improvement) and abs(improvement) <= 1.
        self.past[slot] = (gain+improvement, n+1)

    def install(self, model):
        original = model.mix6

        def mix(base, members, adapted):
            if not adapted or model.recB:
                return original(base, members, adapted)
            assert model.routed and len(model.v5hist) == model.log6['n']
            teacher = np.asarray(model.v5hist[-1], dtype=np.float64)
            assert teacher.shape == (3,) and np.isfinite(teacher).all() and teacher.min() >= 0
            assert teacher.sum() > 0.
            teacher = teacher / teacher.sum()
            edited = {}
            for slot, p in adapted.items():
                scale = self.read(slot)
                frozen = members[model.ft.member[slot]]
                assert frozen.shape == p.shape == (3,)
                self.reads.setdefault(slot, []).append((model.log6['n']-1, scale))
                changed = p if scale == 1. else frozen + scale * (p-frozen)
                self.n_changed += int(not np.array_equal(changed, p))
                edited[slot] = changed
                self.observe(slot, frozen, p, teacher)
            return original(base, members, edited)

        model.mix6 = mix
