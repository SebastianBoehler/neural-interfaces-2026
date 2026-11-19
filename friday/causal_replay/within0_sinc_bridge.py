"""Seal matched within0 shadow targets with common fresh Sinc and FBLight."""

import json

import numpy as np

from bridge_mix import archived_zero, check_prefixes, install
from bridge_orders import pinned_control, seal_prefix_inputs, write_npz, write_receipt
from bridge_owner import BASE, HERE, MEANS, PROTO, ROOT, STACK, guarded_load, owner, pin, sha, stamp
from bridge_recognition import observe_recognition, recording_calibration
from bridge_refit import relabel
from member_inputs import ORDERS, pinned_delivery
from sinc_seed0_bridge import fresh_controls, pack


ORDER = "within0"
OLD = MEANS / "orders_v2/within0/status.json"
FROZEN = MEANS / "within0_frozen_probe_v4/qualification.json"
FRESH = BASE / "ni26_friday_sinc_seed0_fresh_orders_20261118"
ORIGIN = BASE / "ni26_friday_sinc_seed0_fresh_full0_v3_20261118"
FBL = MEANS / "within0_fresh_fbl_v1"
FBL_SOURCE = "aa97a1228d220747684efdb1e1d4446a8673a6208063d03458c0532cd90d2e13"
FBL_OWNER = MEANS / "refit_full0_matrix_v2/members/fbl_s0.json"
FBL_OWNER_SHA = "e5882379eded46837578867ef30fe3ac922aef5292835ab9010a69bdf9c7cbc3"
OLD_SHA = "63353ba2bb3e123e624000370e967bee193f7ebfb94dcffe6688d21e93efc881"
FROZEN_SHA = "43fe6dc888a0cef5fd423976577785b671474be2cfa13668059bbb53839a44eb"
FRESH_RECEIPT = "d9eba90ceb18735cc9e2287cacdf21e15b7cca0a9d8f71821ec5db5bf92d5177"
FRESH_SOURCE = "c5459a46ff4f080a7fde80293040c6b5caece95d66d2c7672bbb73b5507bbd4c"
ORIGIN_RECEIPT = "9e1ebfa70bdd50d979567aac2fbc88a93d09dffdd2eacaae2c9837eb302b0768"
ROUTE = ROOT / "runs/s281_v66_route/within0.json"
ROUTE_SHA = "3a16eb9ce67d8451164f7038946d9753f317f49db8e96a686d5f16d36603925b"


def load_common(p, deliveries, order=ORDER):
    proof = json.loads(FROZEN.read_text())
    receipt = json.loads((FRESH / "frozen_orders.json").read_text())
    origin = json.loads((ORIGIN / "frozen.json").read_text())
    if (proof["held_labels_opened"] or not proof["frozen_inputs_eligible"] or proof["order"] != ORDER
            or proof["delivery_sha256"] != stamp(deliveries[ORDERS.index(ORDER)].astype("<i8")) or not proof["fresh_sinc"]["active"]
            or not proof["fresh_sinc"]["same_state_layout_exact"]
            or proof["fresh_sinc"]["source_bank"] != receipt["bank_sha256"]
            or receipt["source_proof"] != ORIGIN_RECEIPT or receipt["source_sha256"] != FRESH_SOURCE
            or receipt["held_labels_opened"] or origin["held_labels_opened"]):
        raise ValueError("fresh Sinc bank did not qualify for the same within0 source fold")
    np.testing.assert_array_equal(p, deliveries[ORDERS.index(order)])
    path = FRESH / "frozen_orders.npz"
    pin(path, receipt["bank_sha256"])
    which = ORDERS.index(order)
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"], np.arange(2655))
        np.testing.assert_array_equal(z["orders"], ORDERS)
        np.testing.assert_array_equal(z["delivery"], deliveries)
        fresh = z["proba"][which]
    if (fresh.shape != (2655, 3) or not np.isfinite(fresh).all() or (fresh < 0).any()
            or not np.allclose(fresh.sum(1), 1, rtol=0, atol=1e-10)):
        raise ValueError("nonfinite or unnormalized live Sinc input")
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as z:
        subject = z["subject"]
    if set(origin["folds"]) != {s.split("/")[-1] for s in subject}:
        raise ValueError("fresh bank has incomplete fit exclusions")
    for person, fold in origin["folds"].items():
        held = subject == f"Dreyer2026Proteus/{person}"
        if fold["fit_rows_sha256"] != stamp(np.flatnonzero(~held).astype("<i8")):
            raise ValueError(f"{person} source exclusion changed")
        pin(ORIGIN / "folds" / f"{person}.pt", fold["checkpoint_sha256"])
        ref = FRESH / "folds" / f"{person}.json"
        pin(ref, receipt["folds"][person])
        item = json.loads(ref.read_text())
        if (item["person"] != person or item["held_labels_opened"] or item["source_proof"] != ORIGIN_RECEIPT
                or item["checkpoint_sha256"] != fold["checkpoint_sha256"]
                or item["source_rows_sha256"] != fold["fit_rows_sha256"] or item["full0"]["changed"]
                or not 0 <= item["full0"]["max_abs"] <= 2e-5 or not 0 <= item["prefix_max_abs"] <= 2e-5):
            raise ValueError(f"independent delivered Sinc receipt differs for {person}")
        archive = ref.with_suffix(".npz")
        pin(archive, item["output_sha256"])
        with np.load(archive, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["index"], np.flatnonzero(held))
            np.testing.assert_array_equal(z["delivery"][which], p[held[p]])
            np.testing.assert_array_equal(z["proba"][which], fresh[held])
    return fresh, dict(path=str(path), bank_sha256=receipt["bank_sha256"], order=order, source_sha256=FRESH_SOURCE,
                       receipt_sha256=FRESH_RECEIPT, origin_receipt_sha256=ORIGIN_RECEIPT,
                       frozen_qualification_sha256=FROZEN_SHA, folds=origin["folds"])


def run(order=ORDER, prior=OLD, prior_sha=OLD_SHA, route_path=ROUTE, route_sha=ROUTE_SHA,
        fbl_release=FBL, fbl_source=FBL_SOURCE):
    for path, expected in ((prior, prior_sha), (FROZEN, FROZEN_SHA), (route_path, route_sha),
                           (FRESH / "frozen_orders.json", FRESH_RECEIPT),
                           (FRESH / "source.sha256", FRESH_SOURCE), (ORIGIN / "frozen.json", ORIGIN_RECEIPT)):
        pin(path, expected)
    pin(fbl_release / "source.sha256", fbl_source)
    pin(FBL_OWNER, FBL_OWNER_SHA)
    old = json.loads(prior.read_text())
    raw, gate, provenance, pins = pinned_control(order)
    route = json.loads(route_path.read_text())
    if (old["order"] != order or old["held_labels_opened"] or not old["exact_historical_zero"]
            or old["source_calibration_sha256"] != route_sha or route["archive_sha256"] != pins[raw]
            or route["order"] != order or route["bases"][STACK]["routed_sessions"]):
        raise ValueError("within0 historical zero or fixed calibration differs")
    with guarded_load(pins, provenance) as opened:
        ctx = owner(order)
        p = ctx.Pperm
        deliveries = pinned_delivery(ROOT / "runs/shuffle_perms.npz")
        np.testing.assert_array_equal(p, deliveries[ORDERS.index(order)])
        calibration_sha = stamp(recording_calibration(ctx))
        archived = ctx.Z[f"{STACK}|Ps|{PROTO}"].copy()
        original = install(ctx, {})
        zero = install(ctx, {slot: archived[slot] for slot in (1, 3)})
        for tape in (original, zero):
            archived_zero(ctx, tape, raw, gate)
        if any(not np.array_equal(original[k], zero[k]) for k in ("mix", "rebal", "gate", "em", "shadow")):
            raise ValueError("explicit unchanged-EA zero differs from the archive")
        fresh, bank = load_common(p, deliveries, order)
        fbl, fbl_bank = load_fbl(p, order, fbl_release, fbl_source)
        arms, activity = fresh_controls(ctx, archived, fresh, zero, order, fbl=fbl)
        prefix = {arm: check_prefixes(ctx, tape) for arm, tape in {"zero": zero, **arms}.items()}
        bridge = pack(p, arms)
        route_values, recognition = observe_recognition(ctx, bridge, route_path)
        if (recognition["bio_finite_scores"] != 2515 or recognition["calibration_posterior_sha256"] != calibration_sha
                or recognition["bio_prefix_checks"] != 14 or recognition["bio_reset_checks"] != 14
                or recognition["contrast_prefix_checks"] != 28):
            raise ValueError("nonmatching live Bio, contrast or fixed recording calibration")
        staged = dict(order=order, exact_historical_zero=True, prior_zero_sha256=prior_sha,
                      source_sha256=sha(HERE / "source.sha256"), bank=bank, fbl_bank=fbl_bank, activation=activity,
                      exact_incumbent_prefix_checks=prefix, recognition=recognition,
                      held_labels_opened=False, scored=False, targets_sealed=False,
                      matched_refits_allowed=False, terminal_score_allowed=False)
        staged["bridge"] = write_npz(HERE / "bridge.npz", bridge)
        staged["routes"] = write_npz(HERE / "route.npz", dict(index=bridge["index"], delivery=p, **route_values))
        if any(v["routed_sessions"] for v in recognition["routing"].values()):
            staged["status"] = "fresh_route_requires_replay"
        else:
            labels, refits = relabel(ctx, bridge)
            if any((v["refits"], v["past_rows"], v["tsem_active"]) != (240, 24180, 240)
                   for v in refits["arms"].values()):
                raise ValueError("scheduled refit count or genuine TSEM activity differs")
            staged.update(refits=refits, inputs=seal_prefix_inputs(order, HERE, labels), targets_sealed=True,
                          status="shared_fbl_sinc_targets_sealed")
        staged["opened"] = {str(path): digest for path, digest in sorted(opened.items())}
    write_receipt(HERE / "status.json", **staged)
    print(staged["status"], flush=True)

    if not staged["targets_sealed"]:
        raise SystemExit(2)


def load_fbl(p, order=ORDER, release=FBL, source=FBL_SOURCE):
    proof, path = release / "frozen.json", release / "frozen.npz"
    meta = json.loads(proof.read_text())
    fixed = dict(source=source, order=order, rows=len(p), held_labels_opened=False, legacy_warm_parity=False)
    if any(meta[k] != v for k, v in fixed.items()) or any(a.stat().st_mode & 0o222 for a in (proof, path)):
        raise ValueError("shared fresh FBLight receipt or scope differs")
    pin(path, meta["bank"])
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as z:
        people = z["subject"]
    expected = json.loads(FBL_OWNER.read_text())["fit_states"]
    if set(expected) != set(meta["folds"]) or len(expected) != 10:
        raise ValueError("missing all ten shared fresh FBLight source folds")
    for person, f in meta["folds"].items():
        fit = np.flatnonzero(people != f"Dreyer2026Proteus/{person}").astype("<i8")
        if f["fit_rows_sha256"] != expected[person]["fit_rows_sha"] or f["fit_rows_sha256"] != stamp(fit):
            raise ValueError(f"shared FBLight source exclusion changed for {person}")
        if f["checkpoint_sha256"] != expected[person]["sha256"]:
            raise ValueError(f"shared FBLight source state changed for {person}")
        pin(release / "folds" / f"{person}.json", f["receipt"])
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"], np.arange(len(p)))
        np.testing.assert_array_equal(z["delivery"], p)
        values = z["proba"]
    if values.shape != (len(p), 3) or not np.isfinite(values).all() or not np.allclose(values.sum(1), 1, atol=1e-10):
        raise ValueError("invalid same-state FBLight probabilities")
    return values, dict(path=str(path), bank_sha256=meta["bank"], folds=meta["folds"], source=source,
                        receipt_sha256=sha(proof), legacy_warm_parity=False)


if __name__ == "__main__":
    run()
