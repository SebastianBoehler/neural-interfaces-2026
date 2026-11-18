"""Rebuild matched full0 incumbent state with one shared fresh Sinc frozen seed."""
import argparse
import json

import numpy as np

from bridge_dependents import ARCHIVED_ROUTE_SHA
from bridge_mix import archived_zero, check_prefixes, install, prove_activation
from bridge_orders import seal_prefix_inputs, write_npz, write_receipt
from bridge_owner import BASE, GATE, HERE, ORDER, PROTO, RAW, ROOT, STACK, corrected, guarded_load, owner, pin, sha, stamp
from bridge_recognition import REFERENCE, observe_recognition, recording_calibration
from bridge_refit import relabel
from complete_zero import FEAT, HASHES, ROUTING, no_route
from member_inputs import PERMUTATION_SHA256


BANK = BASE / "ni26_friday_sinc_seed0_fresh_full0_20261118/frozen.npz"
SEED0 = ("sinc", 0)
MEMBER = "sc_sinc"


def load_bank(path, receipt_sha, source_sha, p):
    metadata = path.with_suffix(".json")
    pin(metadata, receipt_sha)
    record = json.loads(metadata.read_text())
    pin(path.parent / "source.sha256", source_sha)
    proof = dict(source_sha256=source_sha, held_labels_opened=False, actor="sinc", seed=0, order=ORDER, rows=len(p))
    if any(record[key] != value for key, value in proof.items()):
        raise ValueError("Sinc bank provenance differs from the approved source")
    digests = ("receipt_sha256", "checkpoint_sha256", "predictions_sha256", "fit_rows_sha256")
    with np.load(FEAT, allow_pickle=False) as z:
        subject = z["subject"]
    expected = {f"P{i:02d}" for i in range(1, 11)}
    if set(record["folds"]) != expected:
        raise ValueError("incomplete source-excluded Sinc folds")
    for name, fold in record["folds"].items():
        source = np.flatnonzero(subject != f"Dreyer2026Proteus/{name}").astype("<i8")
        if fold["fit_rows_sha256"] != stamp(source):
            raise ValueError(f"fresh Sinc source exclusion changed for {name}")
        if any(len(fold[key]) != 64 or len(bytes.fromhex(fold[key])) != 32 for key in digests):
            raise ValueError(f"invalid source artifact identity for {name}")
    for key in digests:
        if len({fold[key] for fold in record["folds"].values()}) != len(expected):
            raise ValueError(f"duplicate source artifact across held-person folds: {key}")
    pin(path, record["bank_sha256"])
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z["index"], np.arange(len(p))) or not np.array_equal(z["delivery"], p):
            raise ValueError("fresh Sinc output layout or pinned order differs")
        full0, recording = z["full0"], z["recording"]
    for values in (full0, recording):
        if (values.shape != (len(p), 3) or not np.isfinite(values).all() or (values < 0).any()
                or not np.allclose(values.sum(1), 1, rtol=0, atol=1e-10)):
            raise ValueError("fresh Sinc frozen outputs are invalid")
    for name, fold in record["folds"].items():
        stem = path.parent / "folds" / name
        for suffix, key in ((".json", "receipt_sha256"), (".pt", "checkpoint_sha256"), (".npz", "predictions_sha256")):
            pin(stem.with_suffix(suffix), fold[key])
        held = subject == f"Dreyer2026Proteus/{name}"
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
            if (not np.array_equal(z["index"], np.flatnonzero(held)) or not np.array_equal(z["delivery"], p[held[p]])
                    or not np.array_equal(z["full0"], full0[held]) or not np.array_equal(z["recording"], recording[held])):
                raise ValueError(f"fresh Sinc canonical bank disagrees with its original held fold {name}")
    if np.array_equal(full0, recording):
        raise ValueError("fresh Sinc recording and delivered outputs were reused")
    return full0, dict(bank_sha256=record["bank_sha256"], receipt_sha256=receipt_sha,
                       source_sha256=source_sha, recording_sha256=stamp(recording), folds=record["folds"])


def change_sinc(ctx, archived, frozen, old, anchored_rec):
    before = archived + np.mean([old["sinc", seed] - anchored_rec["sinc", seed] for seed in range(6)], 0)
    next_base = dict(old)
    next_base[SEED0] = frozen
    after = archived + np.mean([next_base["sinc", seed] - anchored_rec["sinc", seed] for seed in range(6)], 0)
    formula = float(np.abs((after - before) - ((frozen - old[SEED0]) / 6)).max())
    if formula > 1e-14:
        raise ValueError("fresh Sinc delta did not have exactly one-sixth weight")
    if float(np.abs(after - before).max()) <= 1e-8:
        raise ValueError("fresh Sinc did not change the frozen mixture")
    ctx.FR17 = next_base
    return ctx.norm(before)[ctx.Pperm], ctx.norm(after)[ctx.Pperm], formula


def fresh_controls(ctx, archived, frozen, zero):
    old = ctx.FR17.copy()
    anchored_rec = ctx.FR6.copy()
    slot = ctx.SLOT["sinc"]
    if slot != 7 or ctx.BS[STACK] != list(range(6)) or ctx.ALLSEED:
        raise ValueError("historical Sinc slot, seed count or active-refit policy changed")
    original, want, error = change_sinc(ctx, archived[slot], frozen, old, anchored_rec)
    if not np.array_equal(zero["members"][MEMBER], original):
        raise ValueError("independent Sinc formula does not reproduce the archived mixer member")
    patches = corrected(ctx.Pperm)
    arms = {"historical": install(ctx, {}), "corrected": install(ctx, patches)}
    for arm, tape in arms.items():
        if not np.array_equal(tape["members"][MEMBER], want):
            raise ValueError(f"fresh Sinc replacement missed the live {arm} mixer")
    if any(not np.array_equal(arms["historical"]["members"][key], value)
           for key, value in zero["members"].items() if key != MEMBER):
        raise ValueError("fresh historical arm changed an unregistered live member")
    if any(ctx.FR17[key] is not v for key, v in old.items() if key != SEED0):
        raise ValueError("an unrelated archived full0 member was substituted")
    if any(ctx.FR6[key] is not v for key, v in anchored_rec.items()):
        raise ValueError("archived recording anchor was modified")
    changed = float(np.abs(arms["historical"]["shadow"] - zero["shadow"]).max())
    if changed <= 1e-8:
        raise ValueError("fresh common Sinc did not reach the live incumbent shadow")
    return arms, dict(sinc_formula_max_abs=error, sinc_slot_delta_max_abs=float(np.abs(want - original).max()),
                      common_sinc_shadow_max_abs=changed, ea_activation=prove_activation(
                          arms["historical"], arms["corrected"], patches))


def pack(p, arms):
    inverse = np.argsort(p)
    views = arms["historical"]["views"]
    result = dict(index=np.arange(len(p)), delivery=p, session=views["sess"][inverse],
                  ge=views["GE"][inverse], pc128=views["pc128"][inverse])
    for arm, tape in arms.items():
        for key in ("mix", "rebal", "gate", "em", "shadow"):
            result[f"{arm}|{key}"] = tape[key][inverse]
    return result


def run(bank, receipt_sha, source_sha):
    zero_path = HERE / "upstream_zero.json"
    zero_sha = "9b9dc4070b234a6df2c2d806c7ad5f2ddf2dbd231370fae688369a78b046663c"
    pin(zero_path, zero_sha)
    known_zero = json.loads(zero_path.read_text())
    if not known_zero["exact_archive_zero"] or known_zero["held_labels_opened"] or known_zero["scored"]:
        raise ValueError("archived terminal zero does not qualify")
    pin(FEAT, HASHES[FEAT])
    pin(ROOT / "runs/shuffle_perms.npz", PERMUTATION_SHA256)
    pin(REFERENCE, ARCHIVED_ROUTE_SHA)
    pin(ROUTING, HASHES[ROUTING])
    no_route()
    with guarded_load() as opened:
        ctx = owner()
        p = ctx.Pperm
        with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as z:
            if not np.array_equal(p, z["full|0|all_loso"]):
                raise ValueError("expected immutable full0 permutation")
        calibration = stamp(recording_calibration(ctx))
        archived = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
        orig = install(ctx, {})
        zero = install(ctx, {k: archived[k] for k in (1, 3)})
        for anchor in (orig, zero):
            archived_zero(ctx, anchor, RAW, GATE)
        if any(not np.array_equal(orig[key], zero[key]) for key in ("mix", "rebal", "gate", "em", "shadow")):
            raise ValueError("explicit unchanged-EA historical zero differs")
        frozen, bank_proof = load_bank(bank, receipt_sha, source_sha, p)
        arms, activation = fresh_controls(ctx, archived, frozen, zero)
        prefix = {arm: check_prefixes(ctx, tape) for arm, tape in arms.items()}
        bridge = pack(p, arms)
        routes, route_proof = observe_recognition(ctx, bridge)
        if route_proof["calibration_posterior_sha256"] != calibration or route_proof["bio_finite_scores"] != 2515:
            raise ValueError("fixed historical calibration or real Bio activity differs")
        routed = any(v["routed_sessions"] for v in route_proof["routing"].values())
        proof = dict(order=ORDER, historical_zero=True, terminal_zero_sha256=zero_sha, source_sha256=sha(HERE / "source.sha256"),
                     bank=bank_proof, activation=activation, exact_incumbent_prefix_checks=prefix,
                     recognition=route_proof, held_labels_opened=False, scored=False,
                     matched_refits_allowed=False, terminal_score_allowed=False)
        proof["bridge"] = write_npz(HERE / "bridge.npz", bridge)
        proof["routes"] = write_npz(HERE / "route.npz", dict(index=bridge["index"], delivery=p, **routes))
        if routed:
            proof["status"] = "fresh_route_requires_replay"
        else:
            labels, refits = relabel(ctx, bridge)
            if any((v["refits"], v["past_rows"]) != (240, 24180) or not 0 < v["tsem_active"] <= 240
                   for v in refits["arms"].values()):
                raise ValueError("fresh Sinc path changed the registered refit schedule")
            proof.update(refits=refits, inputs=seal_prefix_inputs(ORDER, HERE, labels),
                         status="ready_for_matched_refits", matched_refits_allowed=True)
        proof["opened"] = {str(path): digest for path, digest in sorted(opened.items())}
    write_receipt(HERE / "status.json", **proof)
    print(proof["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank_receipt_sha")
    parser.add_argument("bank_source_sha")
    args = parser.parse_args()
    run(BANK, args.bank_receipt_sha, args.bank_source_sha)
