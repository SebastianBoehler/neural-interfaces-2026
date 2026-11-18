"""Verify the historical zero and seal full0 incumbent shadow dependencies."""
from contextlib import contextmanager
import json
import math

import numpy as np

from bridge_owner import (
    GATE, HERE, ORDER, PINS, PROTO, RAW, ROOT, STACK, corrected, guarded_load, owner, sha, stamp,
)

PREFIX = 40


def install(ctx, replacements):
    apply = ctx.S.apply
    mix = ctx.A.mix
    gate = ctx.gm.gate
    em = ctx.gm.em_g
    tape = {}

    def apply_slots(orders):
        if tape or not np.array_equal(orders[PROTO], ctx.Pperm):
            raise ValueError("unexpected member delivery or repeated application")
        apply(orders)
        p = orders[PROTO]
        members = ctx.Z[f"{STACK}|Ps|{PROTO}"]
        for slot, values in replacements.items():
            members[slot] = values[p]
        tape["apply"] = 1

    def observe_mix(bn, proto, fams, extra):
        seen = ctx.A.members(bn, proto)
        if tape.get("apply") != 1 or proto != PROTO or bn != STACK or "mix" in tape:
            raise ValueError("unexpected incumbent mixer path")
        for slot, values in replacements.items():
            if not np.array_equal(seen[slot], values[ctx.Pperm]):
                raise ValueError(f"slot {slot} not consumed after exactly one order index")
        result, share = mix(bn, proto, fams, extra)
        items = [item for fam in fams for item in ctx.A.FAM[fam]] + list(extra)
        used = {key for key, _ in items}
        if len(used) != len(items) or set(seen) - used != set(ctx.SLOT.values()):
            raise ValueError("duplicate or missing live mixer member")
        tape.update(mix=result, share=share, members={key: seen[key].copy() for key, _ in items}, items=items)
        return result, share

    def observe_gate(b, proto):
        tape["rebal"] = b
        tape["gate"] = gate(b, proto)
        return tape["gate"]

    def observe_em(b, proto, g):
        if b is not tape["rebal"] or g is not tape["gate"]:
            raise ValueError("EM input differs from current arm")
        tape["em"] = em(b, proto, g)
        return tape["em"]

    ctx.S.apply, ctx.A.mix, ctx.gm.gate, ctx.gm.em_g = apply_slots, observe_mix, observe_gate, observe_em
    try:
        tape["shadow"] = ctx.chain(STACK, "P0r", ctx.Pperm)
    finally:
        ctx.S.apply, ctx.A.mix, ctx.gm.gate, ctx.gm.em_g = apply, mix, gate, em
    keys = ("sess", "GE", "pc128")
    tape["views"] = {key: ctx.Z[f"{key}|{PROTO}"].copy() for key in keys}
    return tape


@contextmanager
def prefix_views(ctx, views, subset):
    keys = {f"{key}|{PROTO}": value[subset] for key, value in views.items()}
    previous = {key: ctx.Z[key] for key in keys}
    ctx.Z.update(keys)
    try:
        yield
    finally:
        ctx.Z.update(previous)


def stages(ctx, tape, rows):
    with prefix_views(ctx, tape["views"], rows):
        b = ctx.gm.M.rebal(tape["mix"][rows], PROTO, None)
        g = ctx.gm.gate(b, PROTO)
        e = ctx.gm.em_g(b, PROTO, g)
        t = ctx.gm.tsem_g(e, PROTO, g)
    return dict(rebal=b, gate=g, em=e, shadow=t)


def check_prefixes(ctx, tape):
    sess = tape["views"]["sess"]
    sessions = tuple(dict.fromkeys(sess))
    if len(sessions) != 14:
        raise ValueError("unexpected held-new session count")
    by_session = [np.flatnonzero(sess == session)[:PREFIX] for session in sessions]
    for rows in [*reversed(by_session), by_session[0]]:
        check = stages(ctx, tape, rows)
        for key, value in check.items():
            if not np.array_equal(value, tape[key][rows]):
                raise ValueError(f"{key} differs for a truncated interleaved prefix")
    for key in ("mix", "rebal", "em", "shadow"):
        value = tape[key]
        if value.shape != (len(sess), 3) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"invalid live {key} output")
        if not np.allclose(value.sum(1), 1, rtol=0, atol=2e-6):
            raise ValueError(f"{key} rows are not probabilities")
    return len(by_session) + 1


def archived_zero(ctx, tape, raw_path, gate_path):
    inverse = np.argsort(ctx.Pperm)
    with np.load(raw_path) as raw, np.load(gate_path) as gate:
        # The original P0r archive is float32 in recording layout. The gate trace is in delivery layout.
        if (not np.array_equal(tape["shadow"][inverse].astype(np.float32), raw[f"{STACK}|P0r"])
                or not np.array_equal(tape["rebal"], gate[f"{STACK}|input"])
                or not np.array_equal(tape["gate"], gate[f"{STACK}|gate"])):
            raise ValueError("historical incumbent identity failed against pinned outputs")


def prove_activation(zero, active, slots):
    if zero["items"] != active["items"] or zero["share"] != active["share"]:
        raise ValueError("active replay changed the member list or weights")
    if not all(np.array_equal(zero["views"][key], active["views"][key]) for key in zero["views"]):
        raise ValueError("active replay changed the source views")
    weights = dict(zero["items"])
    weight_total = math.fsum(weights.values())
    if any(not np.array_equal(zero["members"][k], active["members"][k]) for k in weights if k not in slots):
        raise ValueError("an unrelated live member changed")
    delta = {slot: active["members"][slot] - zero["members"][slot] for slot in slots}
    if any(float(np.abs(x).max()) <= 1e-3 for x in delta.values()):
        raise ValueError("causal replacement inactive at the mixer")
    expected = sum(weights[slot] * value for slot, value in delta.items()) / weight_total
    observed = active["mix"] - zero["mix"]
    error = float(np.abs(expected - observed).max())
    if error > 1e-14:
        raise ValueError("real mixer differs from two-slot weighted delta")
    diff = {k: float(np.abs(active[k] - zero[k]).max()) for k in ("mix", "rebal", "gate", "em", "shadow")}
    if not all(v > 0 for v in diff.values()):
        raise ValueError("corrected inputs did not activate every live shadow stage")
    return dict(slot_max_abs={slot: float(np.abs(x).max()) for slot, x in delta.items()},
                mixer_formula_max_abs=error, stages_max_abs=diff)


def replay(ctx, order, raw_path, gate_path):
    p = ctx.Pperm
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as z:
        if not np.array_equal(p, z[f"{order[:-1]}|{order[-1]}|all_loso"]):
            raise ValueError("historical path is not the exact pinned order")
    original = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
    baseline = install(ctx, {})
    archived_zero(ctx, baseline, raw_path, gate_path)
    slots = {slot: original[slot] for slot in (1, 3)}
    zero = install(ctx, slots)
    archived_zero(ctx, zero, raw_path, gate_path)
    for key in ("mix", "rebal", "gate", "em", "shadow"):
        if not np.array_equal(zero[key], baseline[key]):
            raise ValueError(f"historical identity bridge changed {key}")
    if any(not np.array_equal(zero["members"][k], baseline["members"][k]) for k in zero["members"]):
        raise ValueError("historical identity bridge changed a member")
    print(f"PASS baseline and archived-slot zero {order} {len(p)} exact", flush=True)
    active = install(ctx, corrected(p, order))
    activation = prove_activation(zero, active, slots)
    prefix = {tag: check_prefixes(ctx, tape) for tag, tape in (("zero", zero), ("corrected", active))}
    inverse = np.argsort(p)
    payload = dict(index=np.arange(len(p)), delivery=p, session=active["views"]["sess"][inverse],
                   ge=active["views"]["GE"][inverse], pc128=active["views"]["pc128"][inverse])
    for name, tape in (("historical", zero), ("corrected", active)):
        for key in ("mix", "rebal", "gate", "em", "shadow"):
            payload[f"{name}|{key}"] = tape[key][inverse]
    checks = dict(activation=activation, exact_prefix_checks=prefix, prefix_len=PREFIX)
    return payload, checks


def main():
    with guarded_load() as opened:
        payload, checks = replay(owner(), ORDER, RAW, GATE)
    results = HERE / "bridge_full0.npz"
    with results.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    receipt = dict(order=ORDER, scope="incumbent unrouted P0r shadow", build=sha(HERE / "source.sha256"),
                   output_sha256=sha(results), reference={str(k): v for k, v in PINS.items()},
                   **checks,
                   held_labels_opened=False, recognition_replayed=False, refit_labels_replayed=False,
                   scored=False, zero=True, delivery_sha256=stamp(payload["delivery"].astype("<i8")),
                   opened={str(path): h for path, h in sorted(opened.items())})
    with (HERE / "bridge_full0.json").open("x") as fd:
        json.dump(receipt, fd, indent=2, sort_keys=True)
        fd.write("\n")
    print(f"PASS causal shared slots, full0 prefix/reset {receipt['output_sha256']}", flush=True)


if __name__ == "__main__":
    main()
