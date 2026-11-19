"""Authenticate the actual saved-state full0 FBLight predictions and replay P01."""

import json

import numpy as np

from refit_smoke import HERE, ROOT, TOL, changed, file_sha
from restore_refit import RELEASE, RELEASE_SHA
from within0_fresh_fbl import evaluate_fold, inputs, scope
from within0_frozen_probe import old_base


ORDER = "full0"
OWNER = "e5882379eded46837578867ef30fe3ac922aef5292835ab9010a69bdf9c7cbc3"


def seal(path, contents):
    with path.open("x") as handle:
        json.dump(contents, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    return file_sha(path)


def main():
    p, states = inputs(ORDER)
    prior = RELEASE / "members/fbl_s0.json"
    if file_sha(prior) != OWNER or file_sha(RELEASE / "source.sha256") != RELEASE_SHA:
        raise ValueError("source controller for full0 frozen FBLight differs")
    controls = json.loads(prior.read_text())
    stored = prior.with_suffix(".npz")
    if (file_sha(stored) != controls["output_sha256"] or controls["fit_states"] != states
            or not 0 <= controls["archived_frozen_max_abs"] <= TOL or "A100" not in controls["gpu"]):
        raise ValueError("invalid authenticated archive of full0 source-fit outputs")
    with np.load(stored, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"], np.arange(len(p)))
        np.testing.assert_array_equal(z["delivery"], p)
        values = z["base"]
    rows, seq, active, prefix, _, spec = evaluate_fold("P01", order=ORDER)
    calibration = changed(active, values[rows])
    if (calibration["changed"] or calibration["max_abs"] > TOL or spec != states["P01"]
            or np.array_equal(rows, seq)):
        raise ValueError(f"real independent source-fit P01 replay differs: {calibration}")
    effect = changed(values, old_base("fbl", np.arange(len(p)), ORDER))
    if (effect["changed"] or not 1e-8 < effect["max_abs"] <= TOL or (values < 0).any()
            or not np.allclose(values.sum(1), 1., rtol=0, atol=1e-10)):
        raise ValueError("the same-source full0 bank is inactive or differs from its frozen archive")
    source, folds = file_sha(HERE / "source.sha256"), {}
    for person, state in states.items():
        held, fit = scope(person)
        ckpt = RELEASE / "checkpoints" / f"fbl_s0_{person}.pt"
        if file_sha(ckpt) != state["sha256"] or fit != state["fit_rows_sha"]:
            raise ValueError(f"source fit or row hash changed for {person}")
        record = dict(order=ORDER, person=person, rows=len(held), owner=OWNER, owner_bank=file_sha(stored),
                      source=source, fit_rows_sha256=fit, checkpoint_sha256=state["sha256"],
                      held_labels_opened=False)
        if person == "P01":
            record["independent_live_prefix"] = dict(posterior=calibration, prefix_max_abs=prefix, exact_reset=True)
        digest = seal(HERE / "folds" / f"{person}.json", record)
        folds[person] = dict(receipt=digest, checkpoint_sha256=state["sha256"], fit_rows_sha256=fit)
    path = HERE / "frozen.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, index=np.arange(len(p)), delivery=p, proba=values)
    path.chmod(0o444)
    proof = dict(source=source, order=ORDER, rows=len(p), bank=file_sha(path), owner=OWNER,
                 held_labels_opened=False, legacy_warm_parity=False, legacy_frozen_comparison=effect, folds=folds)
    print("PASS full0 authenticated frozen bank", seal(HERE / "frozen.json", proof), flush=True)


if __name__ == "__main__":
    main()
