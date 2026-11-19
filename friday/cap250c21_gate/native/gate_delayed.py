"""Exercise the true delayed P10 recognition edge on the archived native CPU scope."""

import json
from pathlib import Path

import numpy as np
import torch

import benchmark_utils.base_solver
from eye_scope import BASE, pin, sha
from live_gate import LiveSelector
from native_eye import prepare
from worker import same_prefix, session


ORIGINAL = BASE / "ni26_friday_return_fbl2_native_v2_20261119"
PIN = "cae74104dce81e0635fd915ef17dc09539ee7125621e06a8b0533c73a4db9ee6"
SMOKE = "35e2888b0563efd0ff2c77e34955eb4b8e078f8815ba0577a1276878ee1c1359"


def run(model, case, rows):
    live = []

    def observe(current):
        gate = LiveSelector(current, audit=True)
        current.predict = gate.predict
        live.append(gate)

        def frame(trace, age):
            np.testing.assert_array_equal(trace["v6"]["mix6"], gate.events[age][2])
            if gate.age != age + 1:
                raise ValueError("live selector was bypassed on the returning path")
        return frame

    result = session(model, "causal", case, rows, observe)
    return result, live[0]


def main():
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("clean owning CPU worker is required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pin(ORIGINAL / "source.sha256", PIN)
    trace = ORIGINAL / "smoke/2/native_fbl2.npz"
    pin(trace.with_suffix(".json"), SMOKE)
    proof = json.loads(trace.with_suffix(".json").read_text())
    pin(trace, proof["output"])
    model, _, cases, gallery = prepare("P10")
    case = cases["2"]
    stock, gate = run(model, case, 31)
    s, h, result = gate.stages()
    with np.load(trace, allow_pickle=False) as old:
        np.testing.assert_array_equal(case["rows"][:31], old["rows"])
        np.testing.assert_array_equal(s[:,2], old["stock_scores"])
        np.testing.assert_array_equal(h[:,2], old["fbl2_scores"])
        np.testing.assert_array_equal(result.argmax(1), old["stock_labels"])
    rec = np.array([r[7] for r in gate.events])
    use = (np.arange(31) >= 20) & ~rec
    np.testing.assert_array_equal(np.flatnonzero(use), np.arange(20,28))
    np.testing.assert_array_equal(result[use], h[use,2])
    np.testing.assert_array_equal(result[~use], s[~use,2])
    activity, hook = stock[2], gate.summary()
    if (hook["recognized"] != 28 or hook["eligible"] != 8 or hook["native_stop"] != 28
            or hook["hook"]["warm_rows"] != 8 or hook["hook"]["fbl_warm_delta"] < .01
            or activity["warm_ages"] != list(range(20,28)) or activity["steps"] != 20
            or not np.any(h[use,2] != s[use,2])):
        raise ValueError("real delayed recognition or warm frame coverage differs")
    run(model, cases["3"], 9)
    short, other = run(model, case, 29)
    same_prefix(stock, short, 29)
    for expected, actual in zip(gate.stages(), other.stages(), strict=True):
        np.testing.assert_array_equal(expected[:29], actual)
    path = Path("delayed.json")
    with path.open("x") as f:
        json.dump(dict(passed=True, source=sha("source.sha256"), archive=SMOKE, gallery=gallery,
                       scope="conditional native CPU P10 visit 2 full0", order="full0", rows=31,
                       exact_archive_stock_fbl2=True, exact_prefix=True, exact_emissions=True, held_labels_opened=False,
                       activity=activity, alternate=hook, protected_rows=23, accepted=np.flatnonzero(use).tolist()),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS true P10 delayed native recognition", sha(path), flush=True)


if __name__ == "__main__":
    main()
