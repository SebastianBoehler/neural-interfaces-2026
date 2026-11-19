"""Run the whole returning stack with exact archived stock and the hard waveform pair."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import benchmark_utils.base_solver
from eye_scope import VISITS, sha
from native_eye import N_PREFIX, N_SMOKE, UNCHANGED, prepare
from pupil_inputs import SOURCE, inputs
from reference_eye import reference
from return_adapter import ADAPTER_SHA, ReturningAdapter
from worker import same_prefix, session


ARMS = ("stock", "identity", "hard")


def run(model, name, arm, case, n, actors, oracle):
    probes = []

    def install(current):
        tape = ReturningAdapter(current, arm, actors, case["x"], oracle)
        probes.append(tape)

        def inspect(trace, t):
            tape.record()
            if len(tape.trace) != t + 1:
                raise ValueError("native protected stage did not advance")
        return inspect

    result = session(model, name, case, n, install)
    tape = probes[0]
    note = tape.summary()
    if (arm != "stock" and note["raw_calls"] != n or
            arm == "hard" and len(tape.pupil) != n or note["warm_rows"] not in (0, len(result[1].warm))):
        raise ValueError("incomplete actual native outer invocation")
    if arm == "hard" and note["warm_rows"] != len(result[1].warm):
        raise ValueError("hard outer route missed an actual warm shift")
    return result, tape, note


def compare(stock, candidate, n):
    before, after = stock[0], candidate[0]
    for key in UNCHANGED:
        if key != "outer":
            np.testing.assert_array_equal(before[0][key], after[0][key])
    same_prefix(before, (before[0], after[1], after[2]), n)
    if before[2] != after[2] or stock[2]["optimizer_prior"] != candidate[2]["optimizer_prior"]:
        raise ValueError("native actual warm counters or loss inputs changed")
    np.testing.assert_array_equal(stock[1].trace, candidate[1].trace)


def main(slot, smoke):
    if (torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized()
            or "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text()):
        raise ValueError("wrong pinned returning CPU runtime")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, day, n = VISITS[slot]
    n = N_SMOKE if smoke else n
    model, _, cases, calibration = prepare(who)
    case, peer = cases[day], cases["3" if day == "2" else "2"]
    paths, oracle, provenance = inputs(who, day, case["rows"])
    _, other_oracle, other_provenance = inputs(who, "3" if day == "2" else "2", peer["rows"])
    if provenance != other_provenance:
        raise ValueError("neighboring session loaded different source weights")
    actors = tuple(torch.jit.load(str(p), map_location="cpu").eval() for p in paths)
    archive, reference_sha = reference(slot)
    archived_zero = run(model, "native_zero", "stock", case, n, (), None)
    result = {arm: run(model, "causal", arm, case, n, actors, oracle) for arm in ARMS}
    with np.load(archive, allow_pickle=False) as d:
        np.testing.assert_array_equal(case["rows"], d["rows"])
        for arm, run_full in (("native_zero", archived_zero), ("causal", result["stock"])):
            for key in ("scores", "labels", "mix", "shadow", "contrast", "identity", "latch"):
                np.testing.assert_array_equal(run_full[0][0][key], d[f"{arm}|{key}"][:n])
    stock, identity, hard = (result[k] for k in ARMS)
    compare(stock, identity, n)
    compare(stock, hard, n)
    same_prefix(stock[0], identity[0], n)
    for arm in ARMS:
        run(model, "causal", arm, peer, 9, actors, other_oracle)
        prefix = run(model, "causal", arm, case, N_PREFIX, actors, oracle)
        same_prefix(result[arm][0], prefix[0], N_PREFIX)
        if arm == "hard":
            np.testing.assert_array_equal(hard[1].pupil[:N_PREFIX], prefix[1].pupil)
    activity = stock[0][2]
    expected = 28 if slot == 2 else 10
    if activity["recognition"] != expected or activity["steps"] != (20 if slot == 2 else 0):
        raise ValueError("conditional native recognition or warm recipe changed")
    x, y = stock[0][0], hard[0][0]
    terminal = float(np.abs(x["scores"] - y["scores"]).max())
    if hard[2]["max_move"] <= .01 or terminal <= 1e-4 or identity[2]["max_move"] != 0:
        raise ValueError("the true outer intervention or literal identity was inactive")
    out = Path("return_c20.npz")
    with out.open("xb") as f:
        np.savez_compressed(f, rows=case["rows"][:n], stock_scores=x["scores"], stock_labels=x["labels"],
                            hard_scores=y["scores"], hard_labels=y["labels"])
    out.chmod(0o444)
    proof = dict(source=sha(Path(__file__).resolve().parent / "source.sha256"), source_actor=SOURCE,
                 outer_adapter=ADAPTER_SHA, order="full0", person=who, visit=day, slot=slot, rows=n, smoke=smoke,
                 source_folds=provenance, upstream_reference=reference_sha, fixed_old_oof_gallery=calibration,
                 entire_target_excluded=True, held_labels_opened=False, accuracy_computed=False,
                 exact_archive_zero=True, native_identity_exact=True, protected_upstream_exact=True,
                 actual_outer_consumption=True, prefix_rows=N_PREFIX, exact_prefix_reset=True,
                 oracle_max_abs=hard[1].peak, terminal_max_abs=terminal, family_max_abs=hard[2]["max_move"],
                 activity=activity, warm_rows=hard[2]["warm_rows"], raw_calls=hard[2]["raw_calls"],
                 changed=int((x["labels"] != y["labels"]).sum()), output=sha(out))
    path = out.with_suffix(".json")
    with path.open("x") as f:
        json.dump(proof, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS exact native returning hard swap", who, day, n, sha(path), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("slot", type=int, choices=range(4))
    cli.add_argument("--smoke", action="store_true")
    a = cli.parse_args()
    main(a.slot, a.smoke)
