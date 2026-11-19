"""Seal the complete fresh Sinc seed-zero frozen output bank."""

import hashlib
import json

import numpy as np

from fresh_sinc import HERE, PEOPLE, ROOT, TOL, file_sha


def main():
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as dataset:
        subjects = dataset["subject"]
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as permutations:
        order = permutations["full|0|all_loso"]
    n = len(subjects)
    joined = {name: np.full((n, 3), np.nan) for name in ("full0", "recording")}
    manifest = file_sha(HERE / "source.sha256")
    folds = {}
    for person in PEOPLE:
        base = HERE / "folds" / person
        path = base.with_suffix(".json")
        meta = json.loads(path.read_text())
        held = subjects == f"Dreyer2026Proteus/{person}"
        rows = np.flatnonzero(held)
        fit_hash = hashlib.sha256(np.flatnonzero(~held).astype("<i8").tobytes()).hexdigest()
        expected = dict(fold=person, seed=0, source_sha256=manifest, fit_rows_sha256=fit_hash,
                        held_labels_opened=False, deterministic_repeat=person == "P01",
                        checkpoint_sha256=file_sha(base.with_suffix(".pt")),
                        predictions_sha256=file_sha(base.with_suffix(".npz")))
        if any(meta[k] != v for k, v in expected.items()) or max(meta["prefix_abs"].values()) > TOL:
            raise ValueError(f"fresh Sinc scope or provenance differs in {person}")
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as bank:
            np.testing.assert_array_equal(bank["index"], rows)
            np.testing.assert_array_equal(bank["delivery"], order[held[order]])
            for name in joined:
                joined[name][rows] = bank[name]
        folds[person] = {k: meta[k] for k in ("fit_rows_sha256", "checkpoint_sha256", "predictions_sha256")}
        folds[person]["receipt_sha256"] = file_sha(path)
    if n != 2655 or any(not np.isfinite(a).all() or not np.allclose(a.sum(1), 1, rtol=0, atol=1e-10)
                        for a in joined.values()):
        raise ValueError("fresh Sinc bank is incomplete")
    bank = HERE / "frozen.npz"
    with bank.open("xb") as handle:
        np.savez_compressed(handle, index=np.arange(n), delivery=order, **joined)
    bank.chmod(0o444)
    receipt = HERE / "frozen.json"
    with receipt.open("x") as handle:
        json.dump(dict(bank_sha256=file_sha(bank), source_sha256=manifest, folds=folds, held_labels_opened=False,
                       rows=n, actor="sinc", seed=0, order="full0"), handle, indent=2, sort_keys=True)
        handle.write("\n")
    receipt.chmod(0o444)
    print("PASS", file_sha(bank), file_sha(receipt), flush=True)


if __name__ == "__main__":
    main()
