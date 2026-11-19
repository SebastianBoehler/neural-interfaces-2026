"""Check owned count views through actual conditional P10 age-28 recognition."""

import json
from pathlib import Path

import numpy as np
import torch

from count_state import verify_count_state
from eye_scope import pin, sha
from gate_delayed import ORIGINAL, PIN, SMOKE, run
from native_eye import prepare
from shared_view import SharedViewRetiringSelector
from worker import same_prefix


def main():
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("authenticated conditional CPU runtime is required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pin(ORIGINAL / "source.sha256", PIN)
    anchor = ORIGINAL / "smoke/2/native_fbl2.npz"
    pin(anchor.with_suffix(".json"), SMOKE)
    pin(anchor, json.loads(anchor.with_suffix(".json").read_text())["output"])
    model, _, cases, gallery = prepare("P10")
    case = cases["2"]
    stock, dual = run(model, case, 31)
    actual, reuse = run(model, case, 31, SharedViewRetiringSelector)
    same_prefix(stock, actual, 31)
    base, alt, wanted = dual.stages()
    a, b, chosen = reuse.stages()
    np.testing.assert_array_equal(a, base)
    np.testing.assert_array_equal(b, alt[:28])
    np.testing.assert_array_equal(chosen, wanted)
    with np.load(anchor, allow_pickle=False) as previous:
        np.testing.assert_array_equal(case["rows"][:31], previous["rows"])
        np.testing.assert_array_equal(a[:,2], previous["stock_scores"])
        np.testing.assert_array_equal(alt[:,2], previous["fbl2_scores"])
        np.testing.assert_array_equal(chosen.argmax(1), previous["stock_labels"])
    info = reuse.summary()
    keys = ("recognized", "retired_at", "eligible", "alternate_rows", "shared_calls", "verified_covariances",
            "count_view_calls", "verified_count_views")
    if (tuple(info[k] for k in keys) != (28,28,8,28,28,28,28,28)
            or info["hook"]["warm_rows"] != 8 or info["alternate_fit_ages"] != [20]
            or stock[2]["steps"] != 20 or stock[2]["warm_ages"] != list(range(20,28))):
        raise ValueError("true delayed recognition, warm replay or count-view coverage differs")
    count = verify_count_state(model, reuse)
    eligible = (np.arange(31)>=20) & ~np.array([e[7] for e in reuse.events])
    np.testing.assert_array_equal(np.flatnonzero(eligible), np.arange(20,28))
    run(model,cases["3"],9,SharedViewRetiringSelector)
    prefix, peer = run(model,case,29,SharedViewRetiringSelector)
    same_prefix(actual,prefix,29)
    for full, short in zip(reuse.stages(),peer.stages(),strict=True):
        np.testing.assert_array_equal(full[:29],short)
    path = Path("delayed.json")
    with path.open("x") as f:
        json.dump(dict(passed=True,source=sha("source.sha256"),anchor=SMOKE,gallery=gallery,
                       scope="conditional native CPU P10 visit 2 full0",rows=31,held_labels_opened=False,
                       alternate=info,activity=stock[2],eligible_ages=np.flatnonzero(eligible).tolist(),
                       exact_archive=True,exact_dual_emission=True,exact_prefix=True,count=count),
                  f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C21gcvw exact native delayed count view",sha(path),flush=True)


if __name__ == "__main__":
    main()
