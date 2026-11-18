"""Publish a complete eight-order member bank only after authenticating every fold."""
import argparse
import json
from pathlib import Path

import numpy as np

from probe import HERE, ORDERS, TOL, deliveries, digest, fingerprint, write_json


def authenticate(stem, held, seed, data, orders):
    path = HERE / f"{stem}_recording.json"
    meta = json.loads(path.read_text())
    test = np.flatnonzero(data["subject"] == f"Dreyer2026Proteus/{held}")
    fit = np.flatnonzero(data["subject"] != f"Dreyer2026Proteus/{held}")
    expected = dict(fold=held, seed=seed, cfg="fbcnet_ea_tf", slot=1, parity=True, held_labels_opened=False,
                    source="c625ecc736a881836a25f85a741ea9147bf698bf",
                    source_manifest=digest(HERE / "source.sha256"), archive_manifest=digest(HERE / "archive.sha256"),
                    fit_rows_sha=fingerprint(fit.astype("<i8")), held_rows_sha=fingerprint(test.astype("<i8")),
                    n_fit=len(fit), n_held=len(test), source_people=sorted(set(data["subject"][fit])),
                    recording_label_changes=0)
    if any(meta[k] != value for k, value in expected.items()) or not 0 <= meta["recording_max_abs"] <= TOL:
        raise ValueError("archive order parity or source roster differs")
    npz = HERE / f"{stem}_orders.npz"
    checkpoint = HERE / f"{stem}_reconstructed.pt"
    if digest(npz) != meta["prediction_sha"] or digest(checkpoint) != meta["reconstructed_checkpoint_sha"]:
        raise ValueError("sealed inference checkpoint or probabilities differ")
    if set(meta["order_checks"]) != set(ORDERS):
        raise ValueError("causal proof is missing a delivery")
    expected_rows = np.stack([orders[name][np.isin(orders[name], test)] for name in ORDERS])
    with np.load(npz, allow_pickle=False) as z:
        if (not np.array_equal(z["index"], test) or tuple(z["orders"]) != ORDERS or
                not np.array_equal(z["delivery"], expected_rows)):
            raise ValueError("sealed predictions differ from the pinned orders")
        values = z["proba"]
    if values.shape != (8, len(test), 3) or not np.isfinite(values).all() or not np.allclose(values.sum(2), 1):
        raise ValueError("invalid probability bank")
    sessions = set(data["stream"][test])
    for name, delivery in zip(ORDERS, expected_rows):
        checks = meta["order_checks"][name]
        if set(checks) != sessions:
            raise ValueError("causal proof misses a held session")
        for session, check in checks.items():
            direct = delivery[data["stream"][delivery] == session][0]
            if (check["first_row"] != direct or not 0 <= check["prefix_posterior_max_abs"] <= TOL
                    or not 0 <= check["first_direct_max_abs"] <= TOL):
                raise ValueError("prefix or first delivered row differs")
    return test, values, dict(receipt=digest(path), predictions=meta["prediction_sha"],
                              checkpoint=meta["reconstructed_checkpoint_sha"], fit_mode=meta["fit_mode"],
                              recording_max_abs=meta["recording_max_abs"])


def main(root):
    with np.load(root / "data/proteus_windows.npz", allow_pickle=False) as z:
        subject, session = z["subject"], z["session"]
    stream = np.char.add(np.char.add(subject, "|"), session)
    data = dict(subject=subject, stream=stream)
    n = len(subject)
    orders = deliveries(root, np.arange(n), n)
    all_proba = np.full((5, 8, n, 3), np.nan, dtype=np.float64)
    proof, failures = {}, {}
    for seed in range(5):
        for person in range(1, 11):
            held = f"P{person:02d}"
            key = f"{held}_seed{seed}"
            try:
                test, values, receipt = authenticate(f"fbcnet_ea_tf_{key}", held, seed, data, orders)
            except (OSError, KeyError, ValueError) as error:
                failures[key] = str(error)
                continue
            all_proba[seed][:, test] = values
            proof[key] = receipt
    complete = not failures and len(proof) == 50 and np.isfinite(all_proba).all()
    result = dict(complete=bool(complete), passes=len(proof), failures=failures, folds=proof,
                  held_labels_opened=False, downstream_scored=False, orders=list(ORDERS))
    if complete:
        output = HERE / "fbcnet_ea_tf_all_loso_orders.npz"
        with output.open("xb") as handle:
            np.savez_compressed(handle, index=np.arange(n), seeds=np.arange(5), orders=np.array(ORDERS),
                                delivery=np.stack(list(orders.values())), proba=all_proba)
        result["bank"] = digest(output)
    write_json(HERE / "matrix_receipt.json", result)
    if not complete:
        raise ValueError(f"incomplete EA member bank: {len(proof)}/50 folds authenticated")
    print(f"PASS EA member bank 50 folds: {result['bank']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    main(parser.parse_args().archive)
