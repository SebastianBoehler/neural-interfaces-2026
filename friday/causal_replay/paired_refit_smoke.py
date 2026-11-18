"""Exercise the same authentic refit on both regenerated full0 target banks."""

import json

import numpy as np

from refit_smoke import HERE, TOL, changed, file_sha, prepare


BRIDGE = HERE.parent / "dependents_full0_v2"
TARGETS = {
    "historical": "5290dd47d3d19523bfc1308db5882fe116780ff102fd757deaa78d2e051399cf",
    "causal": "ca9aef1a8781316e64cf875ccd9c1a7fe722cecb9e71de3a6d753d07312f5cc6",
}


def load_target(arm, rows, shape):
    folder = "corrected" if arm == "causal" else arm
    path = BRIDGE / folder / "chain_v51b_full0.npz"
    return target_table(path, TARGETS[arm], rows, shape)


def target_table(path, expected_sha256, rows, shape):
    if file_sha(path) != expected_sha256:
        raise ValueError(f"wrong regenerated target bank {path}")
    with np.load(path, allow_pickle=False) as bank:
        emitted, relab = bank["all_loso"][rows], bank["relab"][rows]
        ages, grid = bank["pos"][rows], bank["tgrid"]
    np.testing.assert_array_equal(ages, np.arange(len(rows)))
    np.testing.assert_array_equal(grid, 20 + 10 * np.arange(relab.shape[1]))
    valid = (ages[:, None] < grid) & (grid < len(rows))
    np.testing.assert_array_equal(np.isfinite(relab).all(2), valid)
    if not np.isfinite(emitted).all() or not np.allclose(relab[valid].sum(1), 1):
        raise ValueError("invalid regenerated target probabilities")
    target = np.column_stack((emitted, relab.reshape(len(rows), -1), ages))
    if target.shape != shape:
        raise ValueError("legacy refit target layout differs")
    return target


def main():
    model, x, rows, archived, frozen, _, stream, receipt = prepare()
    values = {}
    receipt.update(input_sha256=TARGETS, checks={})
    for arm in TARGETS:
        target = load_target(arm, rows, archived.shape)
        result = stream(model, x, target, 0)
        np.testing.assert_array_equal(result["base"], frozen)
        values[arm] = result["r15c250"]
        np.testing.assert_array_equal(values[arm][:20], frozen[:20])
        prefix = stream(model, x[:80], target[:80], 0)["r15c250"]
        check = changed(prefix, values[arm][:80])
        if check["changed"] or check["max_abs"] > TOL:
            raise ValueError(f"future input changed the {arm} 80-window prefix")
        receipt["checks"][arm] = check
    delta = changed(values["causal"], values["historical"])
    if delta["max_abs"] <= 1e-4:
        raise ValueError("changing upstream targets did not activate the refit")
    receipt["shifted_refit_output"] = delta
    output = HERE / "paired_fbl_seed0_p01_full0.npz"
    order = np.argsort(rows)
    with output.open("xb") as handle:
        np.savez_compressed(handle, index=rows[order], delivery=rows, frozen=frozen[order],
                            historical=values["historical"][order], causal=values["causal"][order])
    receipt["output_sha256"] = file_sha(output)
    with (HERE / "paired_fbl_seed0_p01_full0.json").open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
