"""Authenticate the actual native prefixes and merge four complete blind visits."""

import argparse
import json
from pathlib import Path

import numpy as np

from eye_scope import DATA, DATA_SHA, ORDER_FILE, ORDER_SHA, REPLAY, VISITS, pin, sha
from reference_eye import GALLERY, VISIT_PINS, reference
from verify_fit import artifacts


FIT_SOURCE = "a2566e46004374d08238f4abf2fb9846486910fd2ec173c66cbc4e1a1f6c65dd"
FLAGS = ("exact_native_zero", "exact_native_control", "eye_routeb_evidence_exact",
         "rest_inner_and_outer_mix_exact", "warm_exact", "prefix_reset_exact")


def deliveries():
    pin(DATA, DATA_SHA)
    pin(ORDER_FILE, ORDER_SHA)
    with np.load(DATA, allow_pickle=False) as z:
        people, visits = z["subject"], z["session"]
    with np.load(ORDER_FILE, allow_pickle=False) as z:
        order = z["full|0|seen_session"]
    result = [order[(people[order] == f"Dreyer2026Proteus/{who}") & (visits[order] == visit)]
              for who, visit, _ in VISITS]
    np.testing.assert_array_equal(np.concatenate(result), order)
    return result


def verify(slot, smoke, delivered):
    who, day, n = VISITS[slot]
    expected_rows = delivered[:31] if smoke else delivered
    root = REPLAY / (f"smoke{slot}" if smoke else f"visit{slot}")
    receipt, bank = root / "eye_replay.json", root / "eye_replay.npz"
    proof = json.loads(receipt.read_text())
    fit, fit_smoke = artifacts(who, FIT_SOURCE)
    expected = dict(source=sha(REPLAY / "source.sha256"), order="full0", person=who, visit=day, slot=slot,
                    rows=len(expected_rows), smoke=smoke, held_labels_opened=False, accuracy_computed=False,
                    source_fit=FIT_SOURCE, traces=fit, fit_smokes=fit_smoke, prefix_rows=17,
                    native_ref=VISIT_PINS[slot][0], fixed_old_oof_gallery=GALLERY[who])
    if (len(delivered) != n or any(proof[k] != v for k, v in expected.items())
            or not all(proof[k] for k in FLAGS) or proof["max_eye_log_delta"] <= .15
            or proof["max_terminal_delta"] <= 1e-4 or not 0 <= proof["native_ocular_trace_max_abs"] <= 2e-5
            or any(x.stat().st_mode & 0o222 for x in (receipt, bank))):
        raise ValueError(f"unqualified native Eye {'smoke' if smoke else 'visit'} {slot}")
    pin(bank, proof["output"])
    old, _ = reference(slot)
    with np.load(bank, allow_pickle=False) as z, np.load(old, allow_pickle=False) as anchor:
        np.testing.assert_array_equal(z["rows"], expected_rows)
        for key in ("scores", "labels"):
            np.testing.assert_array_equal(z[f"stock_{key}"], anchor[f"causal|{key}"][:len(expected_rows)])
        for arm in ("stock", "fresh"):
            scores = z[f"{arm}_scores"]
            if scores.shape != (len(expected_rows), 3) or not np.isfinite(scores).all():
                raise ValueError("incomplete native Eye decisions")
            np.testing.assert_array_equal(z[f"{arm}_labels"], scores.argmax(1))
        actual = int(np.count_nonzero(z["stock_labels"] != z["fresh_labels"]))
        if actual != proof["changed"]:
            raise ValueError("read-only decision change count differs")
        data = dict(z)
    return data, dict(receipt=sha(receipt), output=proof["output"], rows=len(expected_rows), changed=actual,
                      activity=proof["activity"], max_terminal_delta=proof["max_terminal_delta"],
                      max_eye_log_delta=proof["max_eye_log_delta"])


def verify_smokes(seq):
    return {slot: verify(slot, True, seq[slot])[1]["receipt"] for slot in (0, 2)}


def collect():
    seq = deliveries()
    smokes = verify_smokes(seq)
    parts, receipts = zip(*(verify(slot, False, rows) for slot, rows in enumerate(seq)), strict=True)
    keys = ("rows", "stock_labels", "stock_scores", "fresh_labels", "fresh_scores")
    combined = {k: np.concatenate([x[k] for x in parts]) for k in keys}
    rows = combined["rows"]
    if len(rows) != 762 or len(np.unique(rows)) != 762:
        raise ValueError("incomplete sealed four-visit Eye bank")
    p = Path("matched_eye.npz")
    with p.open("xb") as f:
        np.savez_compressed(f, **combined)
    p.chmod(0o444)
    receipt = p.with_suffix(".json")
    proof = dict(source=sha(REPLAY / "source.sha256"), order="full0", windows=762,
                 source_fit=FIT_SOURCE, scope="conditional native returning Eye replacement",
                 native_smokes=smokes, parts=receipts, held_labels_opened=False,
                 accuracy_computed=False, output=sha(p))
    with receipt.open("x") as f:
        json.dump(proof, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    receipt.chmod(0o444)
    print("PASS sealed all four complete Eye visits", sha(receipt), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("smoke", "collect"))
    mode = cli.parse_args().mode
    if mode == "smoke":
        print(verify_smokes(deliveries()), flush=True)
    else:
        collect()
