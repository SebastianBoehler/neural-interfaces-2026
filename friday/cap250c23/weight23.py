"""Isolate added adapted evidence from equal frozen weight in the native outer mix."""
from contextlib import contextmanager
import numpy as np

import inputs as full_parent
import upstream as within_parent
from bridge_owner import PROTO, STACK, stamp

ARMS = ("stock", "identity", "FBL2", "frozen2", "FBL3")
EXTRA = "fc23_extra_fbl"
SLOT = "sc_fbl"


def reroute(base, ctx, archive, patches, incumbent, arm):
    positions = ctx.SLOT["fbl"]
    frozen = archive[positions] + np.mean(
        [ctx.FR17["fbl", seed] - ctx.FR6["fbl", seed] for seed in range(6)], axis=0
    )
    delta = ctx.CU17["fbl", 0] - ctx.FT_BASE["fbl", 0]
    cold = ctx.norm(frozen)[ctx.Pperm]
    adapted = ctx.norm(frozen + delta)[ctx.Pperm]
    warm = np.any(adapted != cold, axis=1)
    assert int(warm.sum()) == 2375 and np.abs(adapted - cold).max() > .01
    saved = {k: stamp(v) for k, v in ctx.CU17.items()}
    inputs = {k: stamp(v) for k, v in ctx.FT_BASE.items()}
    shadow = stamp(incumbent["shadow"])
    mix = ctx.A.mix
    rebal, em, tsem = ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g
    proof = {}

    def altered(bn, proto, families, extra):
        assert EXTRA not in ctx.A.EXTRA and not proof and (bn, proto) == (STACK, PROTO)
        assert list(extra).count((SLOT, 1.)) == 1
        np.testing.assert_array_equal(ctx.A.members(bn, proto)[SLOT], adapted)
        stock, native_share = mix(bn, proto, families, extra)
        assert native_share == 1 / 16
        add = 0 if arm == "identity" else 2 if arm == "FBL3" else 1
        value = cold if arm == "frozen2" else adapted
        if arm in {"frozen2", "identity"}:
            # The original adapted member stays present. Only this added unit has different ownership.
            ctx.A.EXTRA[EXTRA] = dict(ctx.A.EXTRA[SLOT], **{PROTO: value})
            try:
                np.testing.assert_array_equal(ctx.A.members(bn, proto)[EXTRA], value)
                changed, share = mix(bn, proto, families, [*extra, (EXTRA, float(add))])
            finally:
                del ctx.A.EXTRA[EXTRA]
        else:
            changed, share = mix(bn, proto, families, [(k, 1. + add if k == SLOT else w) for k, w in extra])
        assert share == 1 / (16 + add)
        np.testing.assert_allclose(changed, (16 * stock + add * value) / (16 + add), rtol=0, atol=2e-14)
        movement = float(np.abs(changed - stock).max())
        assert movement == 0 if arm == "identity" else movement > .001
        proof.update(add=add, share=share, warm=int(warm.sum()), mixture=stamp(changed),
                     extra=stamp(value), adapted=stamp(adapted), movement=movement)
        return changed, share

    def observe(name, function, gate=False):
        def run(values, proto, *args):
            assert name not in proof and proto == PROTO
            if gate:
                np.testing.assert_array_equal(args[0], incumbent["gate"])
            output = function(values, proto, *args)
            proof[name] = stamp(output)
            return output
        return run

    ctx.A.mix = altered
    ctx.gm.M.rebal = observe("rebalance", rebal)
    ctx.gm.em_g = observe("em", em, True)
    ctx.gm.tsem_g = observe("tsem", tsem, True)
    try:
        result, work = base(ctx, archive, patches, incumbent)
    finally:
        ctx.A.mix, ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g = mix, rebal, em, tsem
    assert set(("mixture", "rebalance", "em", "tsem")) <= proof.keys()
    assert saved == {k: stamp(v) for k, v in ctx.CU17.items()}
    assert inputs == {k: stamp(v) for k, v in ctx.FT_BASE.items()}
    assert shadow == stamp(incumbent["shadow"]) and ctx.GB[STACK, "P0r"] is incumbent["rebal"]
    return result, work, proof


@contextmanager
def variants(order):
    owner, prepare = (full_parent, full_parent.full0) if order == "full0" else (within_parent, within_parent.rebuilt)
    base = owner.candidate
    scores, proofs = {}, {}

    def run(ctx, archive, patches, incumbent):
        assert patches and not scores
        stock, work = base(ctx, archive, patches, incumbent)
        scores["stock"] = stock
        for arm in ARMS[1:]:
            prediction, observed, proof = reroute(base, ctx, archive, patches, incumbent, arm)
            assert work == observed
            scores[arm], proofs[arm] = prediction, proof
        np.testing.assert_array_equal(scores["identity"], stock)
        assert len({proofs[a]["mixture"] for a in ARMS[1:]}) == len(proofs)
        assert proofs["frozen2"]["extra"] != proofs["FBL2"]["extra"]
        return stock, work

    owner.candidate = run
    try:
        with prepare() as ctx:
            if order == "full0":
                anchor, zero, parent = ctx["matched"]["stock"], ctx["zero"], ctx["proof"]
            else:
                anchor, zero, parent = ctx["reference"], ctx["archive_zero"], ctx["parent"]
            assert set(scores) == set(ARMS)
            np.testing.assert_array_equal(anchor["raw"], scores["stock"])
            yield dict(raw=scores, reference=anchor, zero=zero, parent=parent, proofs=proofs,
                       **{k: ctx[k] for k in ("count", "final", "windows", "delivery")})
    finally:
        owner.candidate = base
