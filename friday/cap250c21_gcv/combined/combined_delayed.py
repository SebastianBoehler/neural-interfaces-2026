"""Check combined covariance reuse and retirement on the exact late-recognition CPU fixture."""

import json
from pathlib import Path

import numpy as np
import torch

from combined_selector import SharedCovRetiringSelector
from eye_scope import pin, sha
from gate_delayed import ORIGINAL, PIN, SMOKE, run
from native_eye import prepare
from worker import same_prefix


def main():
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("conditional native CPU reference requires the pinned runtime")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pin(ORIGINAL / "source.sha256", PIN)
    anchor = ORIGINAL / "smoke/2/native_fbl2.npz"
    pin(anchor.with_suffix(".json"), SMOKE)
    anchor_receipt = json.loads(anchor.with_suffix(".json").read_text())
    pin(anchor, anchor_receipt["output"])
    model, _, cases, gallery = prepare("P10")
    case = cases["2"]
    full, dual = run(model, case, 31)
    actual, combined = run(model, case, 31, SharedCovRetiringSelector)
    same_prefix(full, actual, 31)
    old, old_alt, wanted = dual.stages()
    stock, alt, selected = combined.stages()
    np.testing.assert_array_equal(old, stock)
    np.testing.assert_array_equal(alt, old_alt[:28])
    np.testing.assert_array_equal(selected, wanted)
    for a, b in zip(combined.model.cp.C[:28], combined.outer.cp.C, strict=True):
        np.testing.assert_array_equal(a, b)
        if np.shares_memory(a, b):
            raise ValueError("alternate state aliases the stock delayed covariance bank")
    with np.load(anchor, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["rows"], case["rows"][:31])
        np.testing.assert_array_equal(z["stock_scores"], stock[:,2])
        np.testing.assert_array_equal(z["fbl2_scores"], old_alt[:,2])
        np.testing.assert_array_equal(z["stock_labels"], selected.argmax(1))
    b = combined.summary()
    if ((b["alternate_rows"], b["retired_at"], b["eligible"], b["shared_calls"], b["verified_covariances"])
            != (28, 28, 8, 28, 28) or b["hook"]["warm_rows"] != 8 or b["alternate_fit_ages"] != [20]
            or full[2]["steps"] != 20 or full[2]["warm_ages"] != list(range(20,28))):
        raise ValueError("actual late recognition, share, or warm scope differs")
    allowed = (np.arange(31) >= 20) & ~np.array([e[7] for e in combined.events])
    np.testing.assert_array_equal(np.flatnonzero(allowed), np.arange(20,28))
    run(model, cases["3"], 9, SharedCovRetiringSelector)
    prefix, short = run(model, case, 29, SharedCovRetiringSelector)
    same_prefix(actual, prefix, 29)
    for complete, partial in zip(combined.stages(), short.stages(), strict=True):
        np.testing.assert_array_equal(complete[:29], partial)
    path = Path("delayed.json")
    with path.open("x") as f:
        json.dump(dict(passed=True, source=sha("source.sha256"), anchor=SMOKE, gallery=gallery,
                       scope="conditional native CPU P10 visit 2 full0", held_labels_opened=False,
                       rows=31, alternate=b, activity=full[2], eligible_ages=np.flatnonzero(allowed).tolist(),
                       exact_dual_emission=True, exact_prelatch_state=True, exact_prefix=True, exact_archive=True),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C21gct exact late native retirement and sharing", sha(path), flush=True)


if __name__ == "__main__":
    main()
