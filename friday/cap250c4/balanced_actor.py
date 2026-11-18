"""Reuse the verified F-C3 worker protocol with independent CE weight policies."""
import sys

import numpy as np
import torch
import external_actor as worker

from balance import adjust, class_mass


def instrument(model, arm):
    ft = model.ft
    assert model.routed == 1 and model.r15 and not model.v6r
    assert ft.on and ft.n_steps == 5 and ft.t0 == 20 and ft.every == 10
    assert len(ft.member) == 4 and ft.cap == 250
    original = ft._refit
    events = []
    gpu = torch.cuda.get_device_name()

    def fit(t, labels, weight):
        assert not model.recB and ft.relabels == ft.refits+1 and len(ft.lab) == t
        current = np.asarray(labels)
        before = np.asarray(weight, dtype=np.float64)
        assert current.shape == before.shape == (t,)
        assert np.isin(current, (0, 1, 2)).all() and np.all(np.isfinite(before) & (before > 0))
        after = adjust(arm, current, before)
        assert np.all(np.isfinite(after) & (after > 0))
        np.testing.assert_allclose(after.sum(), before.sum(), atol=1e-10, rtol=0)
        # Record actual coefficients after the package's float32 weight cast.
        mass0 = class_mass(current, before.astype(np.float32))
        mass1 = class_mass(current, after.astype(np.float32))
        fraction0, fraction1 = mass0/mass0.sum(), mass1/mass1.sum()
        present = mass0 > 0
        unequal = bool(present.sum() > 1 and np.ptp(fraction0[present]) > 1e-7)
        if unequal:
            dominant = np.argmax(mass0)
            shift = fraction1[dominant]-fraction0[dominant]
            assert {'stock': shift == 0, 'primary': shift < 0, 'placebo': shift > 0}[arm]
        events.append(dict(age=t, disagreement=int(np.count_nonzero(np.asarray(ft.lab[:t]) != current)),
            unequal=unequal, changed=int(np.count_nonzero(after.astype(np.float32) != before.astype(np.float32))),
            coefficient_mass_before=fraction0.tolist(), coefficient_mass_after=fraction1.tolist(), gpu=gpu))
        original(t, labels, weight if arm == 'stock' else after)

    ft._refit = fit
    return events, original


if __name__ == '__main__':
    worker.instrument = instrument
    worker.run(int(sys.argv[1]), sys.argv[2], sys.argv[3:] == ['smoke'])
