"""Rebuild both warm choices under the exact shared-state D-EAwcf inputs."""

from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from bridge_mix import archived_zero, install
from bridge_orders import pinned_control
from bridge_owner import PROTO, STACK, corrected, guarded_load, owner, pin, sha
from complete_mix import candidate
from complete_sinc_pair import pair_refits, verify_incumbent
from complete_tail import load_count, load_personalization
from complete_zero import FEAT, read
from member_inputs import ORDERS
from within0_complete import PINS, input_delivery, qualified_zero, three_aux
from within0_pair_inputs import BRIDGE, CAMP, FAMILIES, ORDER, authenticate


PARENT = CAMP / "within0_terminal_shared_v1"
MATRIX = CAMP / "within0_matched_refit_shared_v1"
PARENT_SOURCE = "ad7f624831d9e25ab636a7096a07a6e64999bdf47e7dcea3994e4210b84d8326"
PARENT_RECEIPT = "8dc7da81083bf117a38a17ac4f9a1333929f27b7c41c3cbd87193a4ca22390ae"
PARENT_ARRAYS = "e7d1ec712824f33d8e8237844e9f87d6509e608170beb294316a9f9bb571d59d"
ZERO = "f5f5a2f5ce8974038afa949e10e3368a3f5e38002831fdfb47373992570d3eb4"
MATRIX_SOURCE = "8ecec990613f98dd3267be06fff7ca7551f6044c296813b80a7dce985d226db4"
BRIDGE_SHA = "5874a02fa2cff86daa6d69c50bc7c0696c86f653e3430877c6e85f18d606109b"


class UnreadableWarm:
    def __getitem__(self, key):
        raise RuntimeError(f"FT-off accidentally consumed a warm read for {key}")


def archive_receipt():
    path = PARENT / "matched_within0.json"
    pin(PARENT / "source.sha256", PARENT_SOURCE)
    pin(path, PARENT_RECEIPT)
    item = json.loads(path.read_text())
    assert item["complete"] and item["matches_upstream"] and item["order"] == ORDER
    assert not item["held_labels_opened"] and not item["scored"]
    assert item["source"] == PARENT_SOURCE and item["archive_zero"] == ZERO
    assert item["matrix_source"] == MATRIX_SOURCE and item["bridge"] == BRIDGE_SHA
    assert set(item["refits"]) == set(FAMILIES)
    bank = PARENT / "matched_within0.npz"
    assert item["output"] == dict(path=str(bank), sha256=PARENT_ARRAYS)
    pin(bank, PARENT_ARRAYS)
    qualified_zero(ZERO)
    return item, bank


def frozen_path(ctx, archived, patches, incumbent, arm):
    """Observe the original frozen V6r or the literal V6Fr zero-delta witness."""
    order = ctx.Pperm
    apply, mix, gate = ctx.S.apply, ctx.A.mix, ctx.gm.gate
    previous_members = ctx.A.EXTRA.copy()
    captured = {"apply": 0, "mix": 0, "gate": 0}

    def apply_slots(orders):
        assert captured["apply"] == 0 and np.array_equal(orders[PROTO], order)
        apply(orders)
        values = ctx.Z[f"{STACK}|Ps|{PROTO}"]
        for slot, p in patches.items():
            values[slot] = p[order]
        captured["apply"] += 1

    def observe_mix(bn, proto, families, extra):
        assert captured["apply"] == 1 and captured["mix"] == 0 and (bn, proto) == (STACK, PROTO)
        members = ctx.A.members(bn, proto)
        w = dict([item for family in families for item in ctx.A.FAM[family]] + list(extra))
        assert w["sc_rest"] == 1. and np.array_equal(members["sc_rest"], ctx.RESTR[order])
        for slot, value in patches.items():
            np.testing.assert_array_equal(members[slot], value[order])
        for family, slot in ctx.SLOT.items():
            p = archived[slot] + np.mean([ctx.FR17[family, i] - ctx.FR6[family, i] for i in range(6)], 0)
            assert w[f"sc_{family}"] == 1.
            np.testing.assert_array_equal(members[f"sc_{family}"], ctx.norm(p)[order])
        captured["mix"] += 1
        return mix(bn, proto, families, extra)

    def observe_gate(values, proto):
        assert values is incumbent["rebal"]
        out = gate(values, proto)
        np.testing.assert_array_equal(out, incumbent["gate"])
        captured["gate"] += 1
        return out

    ctx.S.apply, ctx.A.mix, ctx.gm.gate = apply_slots, observe_mix, observe_gate
    ctx.CU17 = UnreadableWarm() if arm == "V6r" else ctx.FT_BASE
    try:
        ctx.chain(STACK, arm, order)
    finally:
        ctx.S.apply, ctx.A.mix, ctx.gm.gate = apply, mix, gate
        ctx.A.EXTRA.clear()
        ctx.A.EXTRA.update(previous_members)
    assert set(captured.values()) == {1}
    return ctx.OUTS[f"{STACK}|{arm}"].astype(np.float64)


@contextmanager
def rebuilt():
    anchor, bank = archive_receipt()
    evidence = authenticate(pin, BRIDGE_SHA)
    raw, gate, provenance, expected = pinned_control(ORDER)
    for path, digest in PINS.items():
        pin(path, digest)
    final = load_personalization()
    with guarded_load(expected, provenance) as opened:
        ctx = owner(ORDER)
        p = ctx.Pperm
        np.testing.assert_array_equal(p, input_delivery())
        archived = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
        archived_zero(ctx, install(ctx, {}), raw, gate)
        f = Path(evidence["bank"]["path"])
        pin(f, evidence["bank"]["bank_sha256"])
        with np.load(f, allow_pickle=False) as source:
            j = ORDERS.index(ORDER)
            np.testing.assert_array_equal(source["delivery"][j], p)
            np.testing.assert_array_equal(source["index"], np.arange(len(p)))
            ctx.FR17["sinc", 0] = source["proba"][j]
        with np.load(evidence["fbl_bank"]["path"], allow_pickle=False) as source:
            np.testing.assert_array_equal(source["delivery"], p)
            np.testing.assert_array_equal(source["index"], np.arange(len(p)))
            ctx.FR17["fbl", 0] = source["proba"]
        with np.load(evidence["bridge"]["path"], allow_pickle=False) as source:
            bridge = dict(source)
        np.testing.assert_array_equal(bridge["index"], np.arange(len(p)))
        np.testing.assert_array_equal(bridge["delivery"], p)
        sessions = bridge["session"]
        initial = np.concatenate([p[sessions[p] == s][:20] for s in dict.fromkeys(sessions[p])])
        assert len(initial) == 280
        banks, proof = pair_refits(MATRIX, MATRIX_SOURCE, ctx, evidence, BRIDGE_SHA, initial, ORDER)
        assert proof == anchor["refits"]
        work, by_visit = {}, {}
        for family in FAMILIES:
            item = json.loads((MATRIX / "members" / f"{family}_s0.json").read_text())
            work[family] = sum(v["causal"]["steps"] for v in item["visits"].values())
            assert work[family] == 2400 and len(item["visits"]) == 14
            for key, values in item["visits"].items():
                by_visit.setdefault(key, {})[family] = values["causal"]["steps"]
        ctx.FT_BASE, ctx.CU17 = banks["base"], banks["causal"]
        three_aux(ctx)
        patches = corrected(p, ORDER)
        tape = install(ctx, patches)
        verify_incumbent(ctx, bridge, "corrected", tape)
        p0 = tape["shadow"][np.argsort(p)]
        on, delta = candidate(ctx, archived, patches, tape)
        assert delta == anchor["active_seed0_deltas"]["corrected"]
        off = frozen_path(ctx, archived, patches, tape, "V6r")
        zero_delta = frozen_path(ctx, archived, patches, tape, "V6Fr")
        np.testing.assert_array_equal(off, zero_delta)
        np.testing.assert_array_equal(on[initial], off[initial])
        movement = float(np.abs(on-off).max())
        assert movement > .001
        count, windows = load_count(ctx), read(FEAT, "X").astype(np.float64)
        with np.load(bank, allow_pickle=False) as matched:
            np.testing.assert_array_equal(matched["delivery"], p)
            np.testing.assert_array_equal(p0, matched["corrected|P0r_posterior"])
            np.testing.assert_array_equal(on, matched["corrected|cap250_raw"])
            reference = {k: matched[f"corrected|cap250_{k}"] for k in ("raw","filtered","terminal")}
        yield dict(raw={True:on, False:off}, count=count, final=final, windows=windows,
                   delivery=p, reference=reference, p0r=p0, ft_work=work, visits=by_visit,
                   first_20_exact=len(initial), off_zero_exact=True, raw_delta=movement,
                   session=sessions, archive_zero=ZERO, parent=PARENT_RECEIPT)
