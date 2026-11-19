"""Verify real calibration consumption and seal all four blind returning visits."""

import argparse
import json
from pathlib import Path

import numpy as np

from eye_cal_scope import CAL, FIT_SOURCE, TARGET
from eye_scope import VISITS, pin, sha
from head_inputs import HEAD_SOURCE
from reference_eye import GALLERY, VISIT_PINS, reference
from verify_native import deliveries


ARMS = ("production", "source", "calibrated")


def verify(slot, smoke, delivered, head):
    who, day, n = VISITS[slot]
    count = 31 if smoke else n
    root = TARGET / (f"smoke{slot}" if smoke else f"visit{slot}")
    receipt, data = root / "eye_cal.json", root / "eye_cal.npz"
    p = json.loads(receipt.read_text())
    scope = dict(order="full0", slot=slot, person=who, visit=day, rows=count, smoke=smoke,
                 source=sha(TARGET / "source.sha256"), held_labels_opened=False, scored=False,
                 source_fit=FIT_SOURCE, head_source=HEAD_SOURCE, head_receipt=head,
                 native_anchors_exact=True, independent_anchor=VISIT_PINS[slot][0], old_oof_gallery=GALLERY[who],
                 inner_warm_rest_outer_exact=True, prefix_reset_exact=True, prefix_rows=17)
    effect = p["activation"]
    if (len(delivered) != n or any(p[k] != v for k, v in scope.items())
            or not 0 < effect["active_windows"] <= count or effect["first_active"] != (28 if slot == 2 else 10)
            or effect["identifiers"] != [who] or effect["evidence_max_abs"] <= .01 or effect["terminal_max_abs"] <= 1e-4
            or not 0 <= effect["blend_max_abs"] <= 1e-14 or not 0 <= p["original_signal_trace_max_abs"] <= 2e-5
            or not 0 <= p["ocular_net_max_abs"] <= 1e-5
            or any(x.stat().st_mode & 0o222 for x in (receipt, data))):
        raise ValueError(f"native Eye calibration {slot} failed authentication")
    pin(data, p["output"])
    anchor, _ = reference(slot)
    with np.load(data, allow_pickle=False) as z, np.load(anchor, allow_pickle=False) as baseline:
        np.testing.assert_array_equal(z["rows"], delivered[:count])
        for arm in ARMS:
            scores, labels = z[f"{arm}_scores"], z[f"{arm}_labels"]
            if scores.shape != (count, 3) or not np.isfinite(scores).all():
                raise ValueError("incomplete native calibration arm")
            np.testing.assert_array_equal(labels, scores.argmax(1))
        for k in ("scores", "labels"):
            np.testing.assert_array_equal(z[f"production_{k}"], baseline[f"causal|{k}"][:count])
            np.testing.assert_array_equal(z[f"source_{k}"][:effect["first_active"]],
                                          z[f"calibrated_{k}"][:effect["first_active"]])
        arrays = dict(z)
    return arrays, dict(receipt=sha(receipt), predictions=p["output"], rows=count, activity=p["activity"],
                        activation=effect)


def smokes(seq):
    pin(CAL / "source.sha256", HEAD_SOURCE)
    head = sha(CAL / "heads.json")
    return head, {i: verify(i, True, seq[i], head)[1]["receipt"] for i in (0, 2)}


def collect():
    seq = deliveries()
    head, qualified = smokes(seq)
    parts, receipts = zip(*(verify(i, False, r, head) for i, r in enumerate(seq)), strict=True)
    keys = ["rows"] + [f"{arm}_{k}" for arm in ARMS for k in ("labels", "scores")]
    bank = {k: np.concatenate([part[k] for part in parts]) for k in keys}
    if len(bank["rows"]) != 762 or len(np.unique(bank["rows"])) != 762:
        raise ValueError("incomplete matched calibration scope")
    output = Path("matched_cal.npz")
    with output.open("xb") as f:
        np.savez_compressed(f, **bank)
    output.chmod(0o444)
    receipt = output.with_suffix(".json")
    p = dict(source=sha(TARGET / "source.sha256"), order="full0", rows=762, head=head, smokes=qualified,
             parts=receipts, accuracy_computed=False, held_labels_opened=False, output=sha(output))
    with receipt.open("x") as f:
        json.dump(p, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    receipt.chmod(0o444)
    print("PASS matched calibration blind seal", sha(receipt), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("smoke", "collect"))
    if cli.parse_args().mode == "smoke":
        print(smokes(deliveries()), flush=True)
    else:
        collect()
