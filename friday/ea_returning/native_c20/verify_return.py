"""Reauthenticate both true prefixes and the complete blind returning release."""

import argparse
import json
from pathlib import Path

import numpy as np

from eye_scope import BASE, VISITS, pin, sha
from pupil_inputs import SOURCE, TOL, inputs
from reference_eye import GALLERY, VISIT_PINS, reference
from verify_native import deliveries


REPLAY = BASE / "ni26_friday_return_c20_native_v2_20261119"
ADAPTER = "278462cc967095ecd9ecca0f03f066716c3717b18858c5c121271e458c609974"
FLAGS = ("entire_target_excluded", "exact_archive_zero", "native_identity_exact", "protected_upstream_exact",
         "actual_outer_consumption", "exact_prefix_reset")


def verify(slot, smoke, delivery):
    who, day, n = VISITS[slot]
    rows = delivery[:31] if smoke else delivery
    root = REPLAY / ("smoke" if smoke else "visit") / str(slot)
    receipt, prediction = root / "return_c20.json", root / "return_c20.npz"
    p = json.loads(receipt.read_text())
    _, _, provenance = inputs(who, day, delivery)
    expected = dict(source=sha(REPLAY / "source.sha256"), source_actor=SOURCE, outer_adapter=ADAPTER, order="full0",
                    person=who, visit=day, slot=slot, rows=len(rows), smoke=smoke, source_folds=provenance,
                    upstream_reference=VISIT_PINS[slot][0], fixed_old_oof_gallery=GALLERY[who], held_labels_opened=False,
                    accuracy_computed=False, prefix_rows=17)
    if (len(delivery) != n or any(p[k] != v for k, v in expected.items()) or not all(p[k] for k in FLAGS)
            or p["raw_calls"] != len(rows) or not 0 <= p["oracle_max_abs"] <= TOL
            or p["family_max_abs"] <= .01 or p["terminal_max_abs"] <= 1e-4 or p["warm_rows"] != (8 if slot == 2 else 0)
            or any(x.stat().st_mode & 0o222 for x in (receipt, prediction))):
        raise ValueError("incorrect complete native F-C20r producer")
    pin(prediction, p["output"])
    anchor, _ = reference(slot)
    with np.load(anchor, allow_pickle=False) as native, np.load(prediction, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["rows"], rows)
        for key in ("scores", "labels"):
            np.testing.assert_array_equal(z[f"stock_{key}"], native[f"causal|{key}"][:len(rows)])
        for arm in ("stock", "hard"):
            values = z[f"{arm}_scores"]
            if values.shape != (len(rows), 3) or not np.isfinite(values).all():
                raise ValueError("protected prediction bank incomplete")
            np.testing.assert_array_equal(values.argmax(1), z[f"{arm}_labels"])
        if int((z["stock_labels"] != z["hard_labels"]).sum()) != p["changed"]:
            raise ValueError("native changed-decision receipt differs")
        arrays = dict(z)
    return arrays, dict(receipt=sha(receipt), prediction=p["output"], rows=len(rows), changed=p["changed"],
                        activity=p["activity"], oracle_max_abs=p["oracle_max_abs"],
                        family_max_abs=p["family_max_abs"], terminal_max_abs=p["terminal_max_abs"],
                        warm_rows=p["warm_rows"])


def smokes(seq):
    return {str(slot): verify(slot, True, seq[slot])[1]["receipt"] for slot in (0, 2)}


def collect():
    seq = deliveries()
    prefixes = smokes(seq)
    chunks, proof = zip(*(verify(i, False, r) for i, r in enumerate(seq)), strict=True)
    tape = {k: np.concatenate([a[k] for a in chunks]) for k in chunks[0]}
    if len(tape["rows"]) != 762 or len(np.unique(tape["rows"])) != 762:
        raise ValueError("the sealed endpoint is missing actual held rows")
    out = Path("matched_c20r.npz")
    with out.open("xb") as f:
        np.savez_compressed(f, **tape)
    out.chmod(0o444)
    rc = out.with_suffix(".json")
    p = dict(source=sha(REPLAY / "source.sha256"), source_actor=SOURCE, windows=762, order="full0", smokes=prefixes,
             parts=proof, held_labels_opened=False, accuracy_computed=False, bank=sha(out))
    with rc.open("x") as f:
        json.dump(p, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    rc.chmod(0o444)
    print("PASS separate complete F-C20r blind seal", sha(rc), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("smoke", "collect"))
    if cli.parse_args().mode == "smoke":
        print(smokes(deliveries()), flush=True)
    else:
        collect()
