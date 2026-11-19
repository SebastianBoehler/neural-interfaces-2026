"""Prove exact alternate retirement on the real P10 visit-2 late latch."""

import json
from pathlib import Path

import numpy as np
import torch

from eye_scope import pin, sha
from gate_delayed import ORIGINAL, PIN, SMOKE, run
from live_gate import RetiringSelector
from native_eye import prepare
from worker import same_prefix


def main():
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("clean owning CPU worker is required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pin(ORIGINAL / "source.sha256", PIN)
    anchor = ORIGINAL / "smoke/2/native_fbl2.npz"
    pin(anchor.with_suffix(".json"), SMOKE)
    p = json.loads(anchor.with_suffix(".json").read_text())
    pin(anchor, p["output"])
    model, _, cases, gallery = prepare("P10")
    case = cases["2"]
    stock, dual = run(model, case, 31)
    actual, retired = run(model, case, 31, RetiringSelector)
    same_prefix(stock, actual, 31)
    original, alternate, wanted = dual.stages()
    same, shortened, chosen = retired.stages()
    np.testing.assert_array_equal(same, original)
    np.testing.assert_array_equal(shortened, alternate[:28])
    np.testing.assert_array_equal(chosen, wanted)
    with np.load(anchor, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["rows"], case["rows"][:31])
        np.testing.assert_array_equal(original[:,2], z["stock_scores"])
        np.testing.assert_array_equal(alternate[:,2], z["fbl2_scores"])
        np.testing.assert_array_equal(chosen.argmax(1), z["stock_labels"])
    info = retired.summary()
    if (info["alternate_rows"] != 28 or info["retired_at"] != 28 or info["eligible"] != 8
            or info["hook"]["warm_rows"] != 8 or info["alternate_fit_ages"] != [20]
            or stock[2]["steps"] != 20):
        raise ValueError("real P10 late-recognition active state differs")
    use = (np.arange(31) >= 20) & ~np.array([r[7] for r in retired.events])
    np.testing.assert_array_equal(np.flatnonzero(use), np.arange(20,28))
    run(model, cases["3"], 9, RetiringSelector)
    prefix, shortened_run = run(model, case, 29, RetiringSelector)
    same_prefix(actual, prefix, 29)
    for full, short in zip(retired.stages(), shortened_run.stages(), strict=True):
        np.testing.assert_array_equal(full[:29], short)
    path = Path("delayed.json")
    with path.open("x") as f:
        json.dump(dict(passed=True, source=sha("source.sha256"), anchor=SMOKE, gallery=gallery,
                       scope="conditional native CPU P10 visit 2 full0", held_labels_opened=False, rows=31,
                       activity=stock[2], alternate=info, accepted=np.flatnonzero(use).tolist(),
                       exact_archive=True, exact_dual_emission=True, exact_prelatch_state=True, exact_prefix=True),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS retirement at authentic late recognition", sha(path), flush=True)


if __name__ == "__main__":
    main()
