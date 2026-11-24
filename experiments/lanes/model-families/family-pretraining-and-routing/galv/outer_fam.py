"""Lane B4: laneH/outer.py with the replaced family chosen by B4_FAMILY (fbraw default, fbl, sinc, fb); every other line unchanged.
The slot-index assertion of the raw original is replaced by the family's own slot; the zero-change, stage-movement and warm-state asserts stay."""
import os

from contextlib import contextmanager

import numpy as np

import inputs as full_parent
import upstream as within_parent
from bridge_owner import PROTO, STACK, stamp


FAMILY = os.environ.get("B4_FAMILY", "fbraw")
SLOT = "sc_" + FAMILY
MASS = 5 / 6


def record(x):
    return {str(k):stamp(v) for k,v in x.items()}


def replace(original, ctx, archive, patches, incumbent, pupils, active):
    perm = ctx.Pperm
    raw_slot = ctx.SLOT[FAMILY]
    assert (raw_slot == 5 or FAMILY != "fbraw") and ctx.BS[STACK] == list(range(6)) and not ctx.ALLSEED
    base = [ctx.FR17[FAMILY,seed] for seed in range(6)]
    other = np.mean(base[1:],axis=0)
    stock = archive[raw_slot] + np.mean([b-ctx.FR6[FAMILY,i] for i,b in enumerate(base)],axis=0)
    stock = stock + (ctx.CU17[FAMILY,0] - ctx.FT_BASE[FAMILY,0])
    proposed = other.copy()
    if pupils is not None:
        assert pupils.shape == stock.shape
        proposed[active] = pupils[active]
    change = MASS*(proposed-other)
    planned = ctx.norm(stock+change)[perm]
    zero = ctx.norm(stock)[perm]
    np.testing.assert_array_equal(planned[~active[perm]], zero[~active[perm]])
    warm_targets = record(ctx.CU17)
    warm_base = record(ctx.FT_BASE)
    shadow, expected_gate = incumbent["shadow"].copy(), incumbent["gate"].copy()
    old_mix, old_rebal, old_em, old_tsem = ctx.A.mix, ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g
    log = {}

    def checked_mix(bn, proto, families, extra):
        assert not log and (bn,proto)==(STACK,PROTO)
        assert list(extra).count((SLOT,1.)) == 1
        original_members = ctx.A.members(bn,proto)
        np.testing.assert_array_equal(original_members[SLOT], zero)
        previous = ctx.A.EXTRA[SLOT][PROTO]
        try:
            ctx.A.EXTRA[SLOT][PROTO] = planned
            members = ctx.A.members(bn,proto)
            assert set(original_members) == set(members)
            for k,v in members.items():
                np.testing.assert_array_equal(v, planned if k == SLOT else original_members[k])
            out = old_mix(bn,proto,families,extra)
        finally:
            ctx.A.EXTRA[SLOT][PROTO] = previous
        log["mix"] = stamp(out[0])
        return out

    def tap(kind, fun, gate=False):
        def checked(inp, proto, *args):
            assert kind not in log and proto == PROTO
            if gate:
                np.testing.assert_array_equal(args[0], expected_gate)
            out = fun(inp,proto,*args)
            log[kind] = stamp(out)
            return out
        return checked

    ctx.A.mix = checked_mix
    ctx.gm.M.rebal = tap("rebalance",old_rebal)
    ctx.gm.em_g = tap("em",old_em,True)
    ctx.gm.tsem_g = tap("tsem",old_tsem,True)
    try:
        scores, deltas = original(ctx,archive,patches,incumbent)
    finally:
        ctx.A.mix, ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g = old_mix, old_rebal, old_em, old_tsem
    assert set(log) == {"mix","rebalance","em","tsem"}
    assert ctx.GB[STACK,"P0r"] is incumbent["rebal"]
    assert record(ctx.CU17)==warm_targets and record(ctx.FT_BASE)==warm_base
    np.testing.assert_array_equal(incumbent["shadow"],shadow)
    np.testing.assert_array_equal(incumbent["gate"],expected_gate)
    return scores,deltas,dict(log, slot_movement=float(np.abs(planned-zero).max()), clipped=int((stock+change<1e-6).any(1).sum()),
                              shadow=stamp(shadow), gate=stamp(expected_gate), refit_states=warm_targets)


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
        for k in ("hard","jitter"):
            assert provenance[k]["slot_movement"] > .01
            assert all(provenance[k][stage] != provenance["identity"][stage]
                       for stage in ("mix","rebalance","em","tsem"))
        return stock,deltas

    owner.candidate = variants
    try:
        with load() as ctx:
            assert set(states) == {"stock","identity","hard","jitter"}
            if order == "full0":
                stock, zero, parent = ctx["matched"]["stock"], ctx["zero"],ctx["proof"]
            else:
                stock, zero, parent = ctx["reference"],ctx["archive_zero"],ctx["parent"]
            np.testing.assert_array_equal(states["stock"],stock["raw"])
            yield dict(raw=states, proof=provenance, reference=stock, zero=zero, parent=parent,
                       **{k:ctx[k] for k in ("count","final","windows","delivery")})
    finally:
        owner.candidate = original
