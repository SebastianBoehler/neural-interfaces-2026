"""Qualify FBLight alone through the complete native returning stack."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import benchmark_utils.base_solver
from eye_scope import VISITS, sha
from native_eye import UNCHANGED, prepare
from observe_fbl2 import ObserveFBLight
from reference_eye import reference
from worker import same_prefix, session


ARMS = ("stock", "identity", "fbl2")


def run(model, mode, arm, case, n):
    probes = []

    def attach(current):
        observer = ObserveFBLight(current, arm)
        probes.append(observer)
        return observer.inspect

    actual = session(model, mode, case, n, attach)
    hook = probes[0]
    summary = hook.summary()
    if (summary["calls"] != n or summary["warm_rows"] != len(actual[1].warm)
            or summary["frozen_rows"] + len(actual[1].warm) != n):
        raise ValueError("incomplete native outer or warm coverage")
    return actual, np.asarray(hook.records), summary


def compare(stock, other, n, identity=False):
    old, new = stock[0], other[0]
    keys = old[0] if identity else tuple(k for k in UNCHANGED if k != "outer")
    for key in keys:
        np.testing.assert_array_equal(old[0][key], new[0][key])
    same_prefix(old, (old[0], new[1], new[2]), n)
    np.testing.assert_array_equal(stock[1], other[1])
    if old[2] != new[2]:
        raise ValueError("underlying native recognition or real warm activity changed")


def prefix_size(slot):
    return 29 if slot == 2 else 17


def replay(slot, smoke):
    if (torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized()
            or "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text()):
        raise ValueError("the pinned native CPU worker is required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, visit, full_n = VISITS[slot]
    n = 31 if smoke else full_n
    model, _, cases, gallery = prepare(who)
    case, other = cases[visit], cases["3" if visit == "2" else "2"]
    archive, provenance = reference(slot)
    zero = run(model, "native_zero", "stock", case, n)
    controls = {arm: run(model, "causal", arm, case, n) for arm in ARMS}
    with np.load(archive, allow_pickle=False) as original:
        np.testing.assert_array_equal(case["rows"], original["rows"])
        for arm, result in (("native_zero", zero), ("causal", controls["stock"])):
            for key in ("scores", "labels", "mix", "shadow", "contrast", "identity", "latch"):
                np.testing.assert_array_equal(result[0][0][key], original[f"{arm}|{key}"][:n])
    stock, identity, candidate = (controls[k] for k in ARMS)
    compare(stock, identity, n, identity=True)
    compare(stock, candidate, n)
    prefix_n = prefix_size(slot)
    for arm in ARMS:
        run(model, "causal", arm, other, 9)
        shorter = run(model, "causal", arm, case, prefix_n)
        same_prefix(controls[arm][0], shorter[0], prefix_n)
        np.testing.assert_array_equal(controls[arm][1][:prefix_n], shorter[1])
    a, b = stock[0][0], candidate[0][0]
    activity = stock[0][2]
    hook = candidate[2]
    terminal = float(np.abs(a["scores"] - b["scores"]).max())
    if (activity["recognition"] != (28 if slot == 2 else 10) or activity["steps"] != (20 if slot == 2 else 0)
            or activity["cp_stopped"] != activity["recognition"] or hook["positive_route_rows"] == 0
            or hook["warm_rows"] != (8 if slot == 2 else 0) or hook["max_mix_delta"] <= 1e-3
            or hook["core_share_max_delta"] <= 1e-6 or terminal <= 1e-4
            or identity[2]["max_mix_delta"] != 0 or identity[2]["core_share_max_delta"] != 0
            or slot == 2 and (hook["fbl_warm_delta"] <= 1e-3 or hook["adapted_mix_max_delta"] <= 1e-3)):
        raise ValueError(f"actual native FBLight or personalizing share did not qualify: {activity}, {hook}")
    out = Path("native_fbl2.npz")
    with out.open("xb") as f:
        np.savez_compressed(f, rows=case["rows"][:n], stock_scores=a["scores"], stock_labels=a["labels"],
                            fbl2_scores=b["scores"], fbl2_labels=b["labels"])
    out.chmod(0o444)
    proof = dict(source=sha(Path(__file__).resolve().parent / "source.sha256"), order="full0", person=who,
                 visit=visit, slot=slot, rows=n, smoke=smoke, native_ref=provenance, fixed_old_oof_gallery=gallery,
                 hook_source=sha(Path(__file__).resolve().parent / "fbl2.py"), activity=activity, hook=hook,
                 exact_archive_zero=True, exact_native_identity=True, protected_upstream_exact=True,
                 active_route_consumption=True, exact_prefix_reset=True, prefix_rows=prefix_n,
                 terminal_max_abs=terminal, changed=int(np.count_nonzero(a["labels"] != b["labels"])),
                 held_labels_opened=False, accuracy_computed=False, output=sha(out))
    path = out.with_suffix(".json")
    with path.open("x") as f:
        json.dump(proof, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS genuine native returning FBLight alone", who, visit, n, sha(path), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("slot", type=int, choices=range(len(VISITS)))
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    replay(args.slot, args.smoke)
