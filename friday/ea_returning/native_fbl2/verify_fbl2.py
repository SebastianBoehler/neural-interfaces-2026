"""Authenticate real FBLight prefixes and seal the complete blind returning bank."""

import argparse
import json
from pathlib import Path

import numpy as np

from eye_scope import BASE, VISITS, pin, sha
from reference_eye import GALLERY, VISIT_PINS, reference
from verify_native import deliveries


REPLAY = BASE / "ni26_friday_return_fbl2_native_v1_20261119"
HOOK = "736a1dbafce94058df803570ca49eb7cbdf55a2b03f5867305ab0c8e35474ea2"
FLAGS = ("exact_archive_zero", "exact_native_identity", "protected_upstream_exact",
         "active_route_consumption", "exact_prefix_reset")
ARMS = ("stock", "fbl2")
BANK = Path("matched_c21r.npz")


def output(slot, smoke):
    return REPLAY / ("smoke" if smoke else "visit") / str(slot) / "native_fbl2.npz"


def verify(slot, smoke, delivered, source):
    who, visit, full_n = VISITS[slot]
    rows = delivered[:31] if smoke else delivered
    bank = output(slot, smoke)
    receipt = bank.with_suffix(".json")
    proof = json.loads(receipt.read_text())
    expected = dict(source=source, order="full0", person=who, visit=visit, slot=slot, rows=len(rows), smoke=smoke,
                    native_ref=VISIT_PINS[slot][0], fixed_old_oof_gallery=GALLERY[who], hook_source=HOOK,
                    prefix_rows=29 if slot == 2 else 17, held_labels_opened=False, accuracy_computed=False)
    activity, hook = proof["activity"], proof["hook"]
    warm = 8 if slot == 2 else 0
    rec = 28 if slot == 2 else 10
    if (len(delivered) != full_n or any(proof[k] != v for k, v in expected.items())
            or not all(proof[k] for k in FLAGS) or any(p.stat().st_mode & 0o222 for p in (bank, receipt))
            or activity["recognition"] != rec or activity["cp_stopped"] != rec
            or activity["steps"] != (20 if slot == 2 else 0)
            or activity["refit_ages"] != ([20] if slot == 2 else [])
            or activity["warm_ages"] != (list(range(20, 28)) if slot == 2 else [])
            or hook["arm"] != "fbl2" or hook["route_share"] != 1/17 or abs(hook["original_share"] - 1/16) > 1e-16
            or hook["calls"] != len(rows) or hook["warm_rows"] != warm or hook["positive_route_rows"] <= 0
            or hook["max_mix_delta"] <= 1e-3 or hook["core_share_max_delta"] <= 1e-6
            or warm and min(hook["fbl_warm_delta"], hook["adapted_mix_max_delta"]) <= 1e-3
            or proof["terminal_max_abs"] <= 1e-4):
        raise ValueError(f"native returning FBLight scope failed for visit {slot}")
    pin(bank, proof["output"])
    archive, _ = reference(slot)
    with np.load(bank, allow_pickle=False) as z, np.load(archive, allow_pickle=False) as old:
        np.testing.assert_array_equal(rows, z["rows"])
        for key in ("labels", "scores"):
            np.testing.assert_array_equal(z[f"stock_{key}"], old[f"causal|{key}"][:len(rows)])
        for arm in ARMS:
            v = z[f"{arm}_scores"]
            if v.shape != (len(rows), 3) or not np.isfinite(v).all():
                raise ValueError(f"invalid complete rows in {arm}")
            np.testing.assert_array_equal(v.argmax(1), z[f"{arm}_labels"])
        changed = int(np.count_nonzero(z["stock_labels"] != z["fbl2_labels"]))
        if changed != proof["changed"]:
            raise ValueError("signed decision tally differs from actual rows")
        arrays = dict(z)
    return arrays, dict(receipt=sha(receipt), bank=proof["output"], rows=len(rows), changed=changed,
                        activity=activity, hook=hook, terminal_max_abs=proof["terminal_max_abs"])


def smokes(seq, source):
    pin(REPLAY / "source.sha256", source)
    pin(REPLAY / "fbl2.py", HOOK)
    return {str(i): verify(i, True, seq[i], source)[1]["receipt"] for i in (0, 2)}


def seal(source):
    seq = deliveries()
    pre = smokes(seq, source)
    parts, proof = zip(*(verify(i, False, rows, source) for i, rows in enumerate(seq)), strict=True)
    all_rows = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    if len(all_rows["rows"]) != 762 or len(np.unique(all_rows["rows"])) != 762:
        raise ValueError("the complete endpoint is missing actual held rows")
    with BANK.open("xb") as f:
        np.savez_compressed(f, **all_rows)
    BANK.chmod(0o444)
    p = BANK.with_suffix(".json")
    with p.open("x") as f:
        json.dump(dict(source=source, hook=HOOK, order="full0", windows=762, smokes=pre, parts=proof,
                       bank=sha(BANK), held_labels_opened=False, accuracy_computed=False),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    p.chmod(0o444)
    print("PASS complete blind F-C21r seal", sha(p), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("smoke", "seal"))
    cli.add_argument("source")
    args = cli.parse_args()
    if args.mode == "seal":
        seal(args.source)
    else:
        print(smokes(deliveries(), args.source), flush=True)
