"""Seal exact legacy basenames for the two regenerated full0 prefix inputs."""
from pathlib import Path
import json
import shutil

import numpy as np

from bridge_owner import BASE, HERE, pin, sha

RESULT = HERE / "relab_corrected_v51b_full0.npz"
RESULT_SHA = "ca9aef1a8781316e64cf875ccd9c1a7fe722cecb9e71de3a6d753d07312f5cc6"
DEPENDENTS = HERE / "dependents_full0.json"
DEPENDENTS_SHA = "bc8d1c4fa910747730f0e7710243263bf5c4478f468fd373035f36d4f61f20c5"
OLD = BASE / "neural-interfaces-2026/runs/labels/relab_v51b_full0.npz"
OLD_SHA = "f4d2e431cd99cb432bfe8d1f53e9bab13214087df4a77888187abb306ec9dfb1"


def compare(actual, expected):
    if actual.shape != expected.shape or not np.array_equal(np.isnan(actual), np.isnan(expected)):
        raise ValueError("archived and regenerated layouts differ")
    valid = np.isfinite(actual).all(-1)
    actual, expected = actual[valid], expected[valid]
    return dict(rows=len(actual), max_abs=float(np.abs(actual - expected).max()),
                decision_changes=int(np.count_nonzero(actual.argmax(1) != expected.argmax(1))))


def main():
    for path, expected in ((RESULT, RESULT_SHA), (DEPENDENTS, DEPENDENTS_SHA), (OLD, OLD_SHA)):
        pin(path, expected)
    audit = json.loads(DEPENDENTS.read_text())
    if audit["refit_output"] != RESULT_SHA or audit["bio_finite_scores"] != 2515:
        raise ValueError("refit or recognition provenance absent")
    with np.load(RESULT, allow_pickle=False) as fresh, np.load(OLD, allow_pickle=False) as old:
        if any(not np.array_equal(fresh[key], old[key]) for key in ("pos", "tgrid")):
            raise ValueError("historical and regenerated refit order or grid differ")
        common = {key: fresh[key] for key in ("index", "delivery", "pos", "tgrid")}
        historical = {key: fresh["historical|" + key] for key in ("all_loso", "relab")}
        check = {key: compare(value, old[key]) for key, value in historical.items()}
    paths = {arm: HERE / arm / "chain_v51b_full0.npz" for arm in ("historical", "corrected")}
    for path in paths.values():
        path.parent.mkdir(exist_ok=False)
    with paths["historical"].open("xb") as handle:
        np.savez_compressed(handle, **common, **historical)
    with RESULT.open("rb") as source, paths["corrected"].open("xb") as target:
        shutil.copyfileobj(source, target)
    pin(paths["corrected"], RESULT_SHA)
    input_manifest = {arm: dict(path=str(path), sha256=sha(path)) for arm, path in paths.items()}
    receipt = HERE / "prefix_ft_inputs.json"
    with receipt.open("x") as handle:
        json.dump(dict(inputs=input_manifest, upstream_refit=RESULT_SHA, upstream_receipt=DEPENDENTS_SHA,
                       original_legacy_archive=OLD_SHA, old_vs_regenerated=check, held_labels_opened=False),
                  handle, indent=2, sort_keys=True)
        handle.write("\n")
    for path in (*paths.values(), receipt):
        path.chmod(0o444)


if __name__ == "__main__":
    main()
