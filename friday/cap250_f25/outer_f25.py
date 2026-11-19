"""Reuse the exact C20 outer slot hook for all frozen F25 pupil families."""
from contextlib import contextmanager

import numpy as np

from outer import full_parent, replace, within_parent
from pins import ARMS, PUPIL_ARMS


@contextmanager
def arms(order, pupils, active):
    # The owning context authenticates the original stock before the downstream reader.
    owner, load = (full_parent,full_parent.full0) if order=="full0" else (within_parent,within_parent.rebuilt)
    original = owner.candidate
    states, provenance = {}, {}

    def variants(ctx, archive, patches, incumbent):
        assert patches and not states
        stock, deltas = original(ctx,archive,patches,incumbent)
        states["stock"] = stock
        for name, p in pupils.items():
            scores, observed, proof = replace(original,ctx,archive,patches,incumbent,p,active)
            assert observed == deltas
            states[name],provenance[name] = scores,proof
        np.testing.assert_array_equal(states["identity"],stock)
        assert provenance["identity"]["slot_movement"] == 0
        for k in PUPIL_ARMS:
            assert provenance[k]["slot_movement"] > .01
            assert all(provenance[k][stage] != provenance["identity"][stage]
                       for stage in ("mix","rebalance","em","tsem"))
        return stock,deltas

    owner.candidate = variants
    try:
        with load() as ctx:
            assert set(states) == set(ARMS)
            if order == "full0":
                stock, zero, parent = ctx["matched"]["stock"], ctx["zero"],ctx["proof"]
            else:
                stock, zero, parent = ctx["reference"],ctx["archive_zero"],ctx["parent"]
            np.testing.assert_array_equal(states["stock"],stock["raw"])
            yield dict(raw=states, proof=provenance, reference=stock, zero=zero, parent=parent,
                       **{k:ctx[k] for k in ("count","final","windows","delivery")})
    finally:
        owner.candidate = original
