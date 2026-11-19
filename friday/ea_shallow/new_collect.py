"""Authenticate all fresh Shallow EA folds and publish the shared member bank."""
import argparse
import json
from pathlib import Path

import numpy as np

from new_fit import HERE, LAYOUT, ORDERS, PEOPLE, SEEDS, TOL, provenance
from probe import deliveries, digest, fingerprint, write_json


def authenticate(held, seed, subject, stream, orders, common):
    key = f"{held}_seed{seed}"
    path = HERE / f"{key}_receipt.json"
    meta = json.loads(path.read_text())
    test = np.flatnonzero(subject == f"Dreyer2026Proteus/{held}")
    source = np.flatnonzero(subject != f"Dreyer2026Proteus/{held}")
    expected = dict(**common, fold=held, seed=seed, deterministic=True,
                    duplicate_fit_bit_exact=(held, seed) == ("P10", 0),
                    fit_rows_sha=fingerprint(source.astype("<i8")), held_rows_sha=fingerprint(test.astype("<i8")),
                    source_people=sorted(set(subject[source])))
    if any(meta[k] != value for k, value in expected.items()):
        raise ValueError("receipt or source exclusion differs")
    checkpoint = HERE / f"{key}_fresh.pt"
    npz = HERE / f"{key}_orders.npz"
    if digest(checkpoint) != meta["checkpoint_sha"] or digest(npz) != meta["predictions_sha"]:
        raise ValueError("sealed checkpoint or posterior hash differs")
    expected_rows = np.stack([orders[name][np.isin(orders[name], test)] for name in LAYOUT])
    with np.load(npz, allow_pickle=False) as z:
        if (not np.array_equal(z["index"], test) or tuple(z["orders"]) != LAYOUT
                or not np.array_equal(z["delivery"], expected_rows)):
            raise ValueError("pinned order or row identity differs")
        values = z["proba"]
    if (values.shape != (len(LAYOUT), len(test), 3) or values.dtype != np.float64
            or not np.isfinite(values).all() or not np.allclose(values.sum(2), 1, rtol=0, atol=1e-12)):
        raise ValueError("invalid posterior layout or values")
    if set(meta["checks"]) != set(LAYOUT):
        raise ValueError("receipt is missing an order")
    for name, rows in zip(LAYOUT, expected_rows):
        checks = meta["checks"][name]
        if set(checks) != set(stream[test]):
            raise ValueError("receipt is missing a held session")
        for session, check in checks.items():
            if (check["first_row"] != rows[stream[rows] == session][0] or check["first_direct_max_abs"] != 0
                    or not check["prefix_exact"] or not check["reset_exact"]
                    or not 0 <= check["batch_max_abs"] <= TOL):
                raise ValueError("session input, prefix, reset or source equivalence failed")
    deltas = {name: float(np.abs(p - values[0]).max()) for name, p in zip(ORDERS, values[1:])}
    if deltas != meta["delta_from_recording"] or max(deltas.values()) <= 1e-4:
        raise ValueError("causal orders reused recording-order probabilities")
    return test, values, dict(receipt=digest(path), checkpoint=meta["checkpoint_sha"],
                              predictions=meta["predictions_sha"], state=meta["state"],
                              source_label_sha=meta["source_label_sha"], source_prior_sha=meta["source_prior_sha"])


def collect(root):
    with np.load(root / "data/proteus_windows.npz", allow_pickle=False) as z:
        subject, session = z["subject"], z["session"]
    stream = np.char.add(np.char.add(subject, "|"), session)
    index = np.arange(len(subject))
    if len(index) != 2655 or set(subject) != {f"Dreyer2026Proteus/{held}" for held in PEOPLE}:
        raise ValueError("unexpected canonical held roster")
    orders = dict(recording=index, **deliveries(root, index, len(index)))
    values = np.full((len(SEEDS), len(LAYOUT), len(index), 3), np.nan)
    common = provenance()
    proof = {}
    for held in PEOPLE:
        person = []
        for seed in SEEDS:
            test, fold, receipt = authenticate(held, seed, subject, stream, orders, common)
            values[seed][:, test] = fold
            proof[f"{held}_seed{seed}"] = receipt
            person.append(receipt)
        if (len({r["source_label_sha"] for r in person}) != 1 or
                len({r["source_prior_sha"] for r in person}) != 1):
            raise ValueError("paired seeds did not use identical source labels or EA prior")
    if len({r["state"] for r in proof.values()}) != len(proof) or not np.isfinite(values).all():
        raise ValueError("some required source-excluded fits are missing or duplicated")

    output = HERE / "shallow_ea_fresh_all_loso_orders.npz"
    with output.open("xb") as handle:
        np.savez_compressed(handle, index=index, seeds=np.array(SEEDS), orders=np.array(ORDERS),
                            delivery=np.stack([orders[name] for name in ORDERS]),
                            proba=values[:, 1:], recording=values[:, 0])
    write_json(HERE / "fresh_bank_receipt.json",
               dict(**common, complete=True, downstream_scored=False, full_dependent_replay_required=True,
                    orders=list(ORDERS), folds=proof, bank=digest(output)))
    print(f"PASS 60 fits and eight causal orders {digest(output)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    collect(parser.parse_args().root)
