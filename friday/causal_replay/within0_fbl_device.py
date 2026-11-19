"""Compare the exact saved FBLight fits across frozen inference devices."""

import argparse
import json
import os
from time import perf_counter

import numpy as np
import torch

from refit_smoke import HERE, changed, file_sha
from within0_fresh_fbl import QUAL, evaluate_fold, inputs, load_fold


RTX = HERE.parent / "within0_fresh_fbl_v1"
RTX_SOURCE = "aa97a1228d220747684efdb1e1d4446a8673a6208063d03458c0532cd90d2e13"
P01 = "fb047f2d5c67a79005315ee9ad5c58d598c4ef1ab40f8936695500835b8bf95f"
P04 = HERE.parent / "within0_frozen_probe_v3/p04.json"
P04_RTX = "f8800dd69fe619da92a6fa2612945ea7db69e6243684cac99c299673b74a06e7"
DEV_TOL = 3e-6


def rtx_reference(delivery, states):
    if (file_sha(RTX / "source.sha256") != RTX_SOURCE
            or file_sha(RTX / "folds/P01.json") != P01 or file_sha(P04) != P04_RTX):
        raise ValueError("same-state RTX reference receipt changed")
    rows, values, _ = load_fold("P01", delivery, states["P01"], release=RTX)
    proof = json.loads(P04.read_text())
    q = json.loads(QUAL.read_text())
    fold = proof["candidates"]["fbl"]
    if (q["receipts"]["P04"] != P04_RTX or proof["person"] != "P04" or proof["rows"] != 566
            or proof["session_count"] != 3 or proof["held_labels_opened"] or proof["order"] != "within0"
            or not fold["eligible"] or fold["changed"] or not fold["interleaved_reset_exact"]
            or "RTX 2080 Ti" not in proof["gpu"] or proof["torch"] != "2.14.1+cu126"
            or fold["source_checkpoint"] != states["P04"]["sha256"]
            or fold["fit_rows_sha256"] != states["P04"]["fit_rows_sha"]):
        raise ValueError("P04 reference is not the same authenticated RTX state")
    return rows, values, fold["max_abs"]


def run(device):
    p, states = inputs()
    rtx_rows, rtx_values, rtx_p04_to_archive = rtx_reference(p, states)
    results = {}
    for person in ("P01", "P04"):
        start = perf_counter()
        rows, sequence, result, prefix, legacy, spec = evaluate_fold(person, device)
        elapsed = perf_counter() - start
        if person == "P01":
            np.testing.assert_array_equal(rows, rtx_rows)
            diff = changed(result, rtx_values)
            comparison = dict(kind="direct_same_state", **diff, cross_device_bound=diff["max_abs"])
        else:
            comparison = dict(kind="same_state_triangle", **legacy, rtx_to_archive=rtx_p04_to_archive,
                              cross_device_bound=legacy["max_abs"] + rtx_p04_to_archive)
        results[person] = dict(rows=len(rows), active_delivery=not np.array_equal(rows, sequence),
                               checkpoint=spec["sha256"], fit_rows=spec["fit_rows_sha"],
                               exact_first_window=True, exact_reset=True, unchanged_weights=True,
                               prefix_max_abs=prefix, seconds=elapsed, **comparison)
    passed = all(r["active_delivery"] and not r["changed"] and r["cross_device_bound"] <= DEV_TOL
                 for r in results.values())
    receipt = dict(qualified=passed, device=device, runtime=str(torch.__version__),
                   gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                   node=os.environ["SLURMD_NODENAME"], threads=torch.get_num_threads(),
                   held_labels_opened=False, source=file_sha(HERE / "source.sha256"),
                   order="within0", rtx={"P01": P01, "P04": P04_RTX}, bound=DEV_TOL, folds=results)
    out = HERE / f"{device}.json"
    with out.open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    out.chmod(0o444)
    print("PASS" if passed else "REFUTED", device, file_sha(out), results, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=("a100", "cpu"))
    run(parser.parse_args().device)
