"""Replay unchanged Sinc source checkpoints in every pinned held-new order."""

import argparse
import hashlib
import json

import numpy as np
import torch

from fresh_sinc import PEOPLE, replay, state_hash
from member_inputs import ORDERS
from refit_smoke import HERE, ROOT, TOL, changed, file_sha, refit_api, waveform_data
from restore_refit import restore


SOURCE = ROOT.parent / "ni26_friday_sinc_seed0_fresh_full0_v3_20261118"
SOURCE_PROOF = "9e1ebfa70bdd50d979567aac2fbc88a93d09dffdd2eacaae2c9837eb302b0768"


def references():
    if file_sha(SOURCE / "frozen.json") != SOURCE_PROOF:
        raise ValueError("unexpected fresh Sinc source proof")
    proof = json.loads((SOURCE / "frozen.json").read_text())
    if proof["held_labels_opened"] or file_sha(SOURCE / "frozen.npz") != proof["bank_sha256"]:
        raise ValueError("fresh Sinc full0 bank differs")
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as z:
        deliveries = np.stack([z[f"{k[:-1]}|{k[-1]}|all_loso"] for k in ORDERS])
    return proof, deliveries


def replay_fold(person):
    api, _, _ = refit_api()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    proof, deliveries = references()
    data = waveform_data()
    checkpoint = SOURCE / "folds" / f"{person}.pt"
    fold = proof["folds"][person]
    model, fit, held = restore(api, data, "sinc", person, checkpoint, fold["checkpoint_sha256"])
    st = state_hash(model.net)
    source = hashlib.sha256(fit.astype("<i8").tobytes()).hexdigest()
    if source != fold["fit_rows_sha256"]:
        raise ValueError("Sinc fit mask differs")
    rows = np.flatnonzero(held)
    orders = np.stack([q[held[q]] for q in deliveries])
    values = [replay(model, data, rows, sequence) for sequence in orders]
    with np.load(SOURCE / "frozen.npz", allow_pickle=False) as z:
        difference = changed(values[ORDERS.index("full0")][0], z["full0"][rows])
    if difference["changed"] or difference["max_abs"] > TOL or state_hash(model.net) != st:
        raise ValueError("restored Sinc full0 parity or fixed state failed")
    stem = HERE / "folds" / person
    with stem.with_suffix(".npz").open("xb") as handle:
        np.savez_compressed(handle, index=rows, delivery=orders, proba=np.stack([v for v, _ in values]))
    with stem.with_suffix(".json").open("x") as handle:
        json.dump(dict(person=person, source_proof=SOURCE_PROOF, held_labels_opened=False,
                       rows=len(rows), prefix_max_abs=max(e for _, e in values), full0=difference,
                       checkpoint_sha256=fold["checkpoint_sha256"], source_rows_sha256=source,
                       output_sha256=file_sha(stem.with_suffix(".npz"))), handle, indent=2, sort_keys=True)
        handle.write("\n")


def collect():
    proof, deliveries = references()
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as data:
        subject = data["subject"]
    out = np.full((len(ORDERS), len(subject), 3), np.nan)
    checks = {}
    for person in PEOPLE:
        held = subject == f"Dreyer2026Proteus/{person}"
        stem = HERE / "folds" / person
        meta = json.loads(stem.with_suffix(".json").read_text())
        if (meta["person"] != person or meta["source_proof"] != SOURCE_PROOF or meta["held_labels_opened"]
                or meta["checkpoint_sha256"] != proof["folds"][person]["checkpoint_sha256"]
                or meta["source_rows_sha256"] != proof["folds"][person]["fit_rows_sha256"]
                or meta["prefix_max_abs"] > TOL or meta["full0"]["changed"] or meta["full0"]["max_abs"] > TOL):
            raise ValueError(f"unverified {person} held fold")
        path = stem.with_suffix(".npz")
        if file_sha(path) != meta["output_sha256"]:
            raise ValueError("changed replay output")
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["index"], np.flatnonzero(held))
            np.testing.assert_array_equal(z["delivery"], np.stack([q[held[q]] for q in deliveries]))
            out[:, held] = z["proba"]
        checks[person] = file_sha(stem.with_suffix(".json"))
    if out.shape != (8, 2655, 3) or not np.isfinite(out).all() or not np.allclose(out.sum(2), 1, atol=1e-10, rtol=0):
        raise ValueError("incomplete canonical Sinc banks")
    target = HERE / "frozen_orders.npz"
    with target.open("xb") as handle:
        np.savez_compressed(handle, index=np.arange(2655), orders=ORDERS, delivery=deliveries, proba=out)
    target.chmod(0o444)
    receipt = HERE / "frozen_orders.json"
    with receipt.open("x") as handle:
        json.dump(dict(bank_sha256=file_sha(target), source_proof=SOURCE_PROOF, folds=checks,
                       orders=ORDERS, held_labels_opened=False, source_sha256=file_sha(HERE / "source.sha256")),
                  handle, indent=2, sort_keys=True)
        handle.write("\n")
    receipt.chmod(0o444)
    print("PASS", file_sha(target), file_sha(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("person", choices=(*PEOPLE, "collect"))
    args = parser.parse_args()
    collect() if args.person == "collect" else replay_fold(args.person)
