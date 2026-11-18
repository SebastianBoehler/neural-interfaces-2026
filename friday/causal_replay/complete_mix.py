"""Verify SessionBN adaptation and corrected EA inputs at the actual §430 mixer."""
import numpy as np

from bridge_owner import PROTO, STACK


def candidate(ctx, archived, replacements, incumbent):
    if ctx.ALLSEED or not ctx.GATE_P0 or set(ctx.SLOT) != {"fbraw", "fb", "sinc", "fbl"}:
        raise ValueError("the archived seed-0 delta or incumbent gate is inactive")
    if ctx.GB[STACK, "P0r"] is not incumbent["rebal"]:
        raise ValueError("the live P0r gate input has changed")
    p = ctx.Pperm
    planned = {}
    for family, slot in ctx.SLOT.items():
        frozen = archived[slot] + np.mean([ctx.FR17[family, k] - ctx.FR6[family, k] for k in range(6)], 0)
        delta = ctx.CU17[family, 0] - ctx.FT_BASE[family, 0]
        planned[family] = (ctx.norm(frozen + delta)[p], ctx.norm(frozen)[p],
                           float(np.abs(delta).max()))
    apply, mix, gate = ctx.S.apply, ctx.A.mix, ctx.gm.gate
    previous_members = ctx.A.EXTRA.copy()
    captured = {}

    def apply_slots(orders):
        if "applied" in captured or not np.array_equal(orders[PROTO], p):
            raise ValueError("candidate delivery changed or applied twice")
        apply(orders)
        values = ctx.Z[f"{STACK}|Ps|{PROTO}"]
        for slot, corrected in replacements.items():
            values[slot] = corrected[p]
        captured["applied"] = True

    def observe_mix(bn, proto, families, extra):
        if not captured.get("applied") or proto != PROTO or bn != STACK or "deltas" in captured:
            raise ValueError("unexpected candidate mixer")
        members = ctx.A.members(bn, proto)
        weights = dict([item for family in families for item in ctx.A.FAM[family]] + list(extra))
        if weights["sc_rest"] != 1.0 or not np.array_equal(members["sc_rest"], ctx.RESTR[p]):
            raise ValueError("live count path did not consume the matched rest mean")
        for slot in (1, 3):
            wanted = replacements[slot] if slot in replacements else archived[slot]
            if not np.array_equal(members[slot], wanted[p]):
                raise ValueError(f"EA slot {slot} did not reach the candidate exactly once")
        for family, (value, zero, delta) in planned.items():
            key = f"sc_{family}"
            if (weights[key] != 1.0 or not np.array_equal(members[key], value)
                    or np.array_equal(value, zero) or delta <= 1e-3):
                raise ValueError(f"archived six-seed base and seed-0 refit not active in {family}")
        captured["deltas"] = {family: item[2] for family, item in planned.items()}
        return mix(bn, proto, families, extra)

    def observe_gate(values, proto):
        if values is not incumbent["rebal"]:
            raise ValueError("cap250 did not read its own incumbent rebalancing output")
        observed = gate(values, proto)
        if not np.array_equal(observed, incumbent["gate"]):
            raise ValueError("cap250 gate differs from the live incumbent")
        return observed

    ctx.S.apply, ctx.A.mix, ctx.gm.gate = apply_slots, observe_mix, observe_gate
    try:
        ctx.chain(STACK, "V6Fr", ctx.Pperm)
    finally:
        ctx.S.apply, ctx.A.mix, ctx.gm.gate = apply, mix, gate
        ctx.A.EXTRA.clear()
        ctx.A.EXTRA.update(previous_members)
    return ctx.OUTS[f"{STACK}|V6Fr"].astype(np.float64), captured["deltas"]
