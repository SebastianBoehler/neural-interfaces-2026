"""Qualify saved source-excluded frozen fits against the original within0 LPC bases."""

import argparse
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from fresh_sinc import evaluate, replay, state_hash
from member_inputs import ORDERS, pinned_delivery
from refit_smoke import ARCHIVE, HERE, ROOT, TOL, changed, file_sha, waveform_data
from restore_refit import RELEASE, RELEASE_SHA, restore


PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
ORDER = "within0"
ARCHS = {
    "fbraw": ("fbcnet_raw_sbn", 20.0),
    "fb": ("fb_sbn_rot05", 2.0),
    "fbl": ("fblight_ea_rot05_sbn", 5.0),
}
FIT_GPU = {"fbraw": "RTX 2080 Ti", "fb": "RTX 2080 Ti", "fbl": "A100"}
FRESH = ROOT.parent / "ni26_friday_sinc_seed0_fresh_orders_20261118"
FRESH_SOURCE = ROOT.parent / "ni26_friday_sinc_seed0_fresh_full0_v3_20261118"


def old_base(family, rows, order=ORDER):
    suffix = f"ttt18{order}LPC_{family}_base-all_loso-s0.npz"
    with np.load(ARCHIVE / suffix, allow_pickle=False) as bank:
        np.testing.assert_array_equal(bank["index"], np.arange(2655))
        p = bank["proba"]
    if p.shape != (2655, 3) or not np.isfinite(p).all():
        raise ValueError(f"invalid archived base for {family}")
    return p[rows]


def saved_fit(api, family, person, data, rows, sequence, reset_rows, fit_sha):
    meta = json.loads((RELEASE / "members" / f"{family}_s0.json").read_text())
    if ((meta["member"], meta["seed"], meta["order"], meta["rows"], meta["smoke"],
         meta["held_labels_opened"]) != (family, 0, "full0", 2655, False, False)
            or len(meta["fit_states"]) != 10 or FIT_GPU[family] not in meta["gpu"]
            or meta["archived_frozen_max_abs"] > TOL):
        raise ValueError(f"nonmatching source fit receipt for {family}")
    entry = meta["fit_states"][person]
    model, fit, held = restore(api, data, family, person,
                               RELEASE / "checkpoints" / f"{family}_s0_{person}.pt", entry["sha256"])
    np.testing.assert_array_equal(np.flatnonzero(held), rows)
    if fit_sha != rows_sha(fit) or fit_sha != entry["fit_rows_sha"] or len(fit) != entry["n_fit"]:
        raise ValueError(f"source exclusion differs for {family}")
    state = state_hash(model.net)
    for session in np.unique(data["stream"][rows]):
        r = sequence[data["stream"][sequence] == session]
        np.testing.assert_array_equal(model._aligned(r, True)[:1], model._aligned(r[:1], True))
    with torch.inference_mode():
        values, prefix = replay(model, data, rows, sequence)
        for session in np.unique(data["stream"][rows]):
            session_prefix = sequence[data["stream"][sequence] == session][:64]
            before = evaluate(model, session_prefix)
            evaluate(model, reset_rows)
            np.testing.assert_array_equal(evaluate(model, session_prefix), before)
    if state_hash(model.net) != state:
        raise ValueError(f"frozen state changed for {family}")
    difference = changed(values, old_base(family, rows))
    return dict(**difference, eligible=difference["changed"] == 0 and difference["max_abs"] <= TOL,
                prefix_max_abs=prefix, interleaved_reset_exact=True,
                fit_rows_sha256=fit_sha, source_checkpoint=entry["sha256"],
                source_gpu=meta["gpu"])


def fresh_sinc(person, rows, sequence, deliveries, fit_sha):
    manifest = json.loads((FRESH / "frozen_orders.json").read_text())
    producer = json.loads((FRESH_SOURCE / "frozen.json").read_text())
    fold_path = FRESH / "folds" / f"{person}.json"
    fold = json.loads(fold_path.read_text())
    origin = producer["folds"][person]
    if (manifest["held_labels_opened"] or producer["held_labels_opened"] or fold["held_labels_opened"]
            or manifest["orders"] != list(ORDERS) or manifest["source_proof"] != file_sha(FRESH_SOURCE / "frozen.json")
            or manifest["folds"][person] != file_sha(fold_path)
            or fold["source_proof"] != manifest["source_proof"] or fold["person"] != person
            or fold["checkpoint_sha256"] != origin["checkpoint_sha256"]
            or fold["source_rows_sha256"] != origin["fit_rows_sha256"]
            or fold["source_rows_sha256"] != fit_sha
            or fold["prefix_max_abs"] > TOL or fold["full0"]["changed"] or fold["full0"]["max_abs"] > TOL):
        raise ValueError("fresh Sinc proof differs")
    i = ORDERS.index(ORDER)
    with np.load(FRESH / "frozen_orders.npz", allow_pickle=False) as bank:
        np.testing.assert_array_equal(bank["index"], np.arange(2655))
        np.testing.assert_array_equal(bank["delivery"], deliveries)
        np.testing.assert_array_equal(bank["orders"], ORDERS)
        values = bank["proba"][i, rows]
    fold_bank = fold_path.with_suffix(".npz")
    if fold["output_sha256"] != file_sha(fold_bank):
        raise ValueError("fresh Sinc fold output changed")
    with np.load(fold_bank, allow_pickle=False) as direct:
        np.testing.assert_array_equal(direct["index"], rows)
        np.testing.assert_array_equal(direct["delivery"][i], sequence)
        np.testing.assert_array_equal(direct["proba"][i], values)
    difference = changed(values, old_base("sinc", rows))
    return dict(**difference, active=difference["max_abs"] > 1e-8, source_checkpoint=fold["checkpoint_sha256"],
                source_bank=manifest["bank_sha256"], prefix_max_abs=fold["prefix_max_abs"])


def rows_sha(rows):
    return hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()


def main(person):
    from session_bn import make

    gpu = torch.cuda.get_device_name()
    if ("RTX 2080 Ti" not in gpu or torch.__version__ != "2.14.1+cu126"
            or os.environ["SLURM_CPUS_PER_TASK"] != "8" or file_sha(RELEASE / "source.sha256") != RELEASE_SHA):
        raise ValueError("source release or measured GPU runtime differs")
    torch.set_num_threads(8)
    data = waveform_data()
    held = data["subject"] == f"Dreyer2026Proteus/{person}"
    rows = np.flatnonzero(held)
    streams = np.unique(data["stream"][rows])
    if not len(rows) or len(np.unique(data["subject"][~held])) != 9:
        raise ValueError("unexpected held scope")
    fit_sha = rows_sha(np.flatnonzero(~held))
    deliveries = pinned_delivery(ROOT / "runs/shuffle_perms.npz")
    p = deliveries[ORDERS.index(ORDER)]
    np.testing.assert_array_equal(np.sort(p), np.arange(2655))
    with np.load(ROOT.parent / "ni26_friday_causal_rebuild_20261118/orders_v2/within0/bridge.npz",
                 allow_pickle=False) as bridge:
        np.testing.assert_array_equal(bridge["delivery"], p)
    sequence = p[held[p]]
    np.testing.assert_array_equal(np.sort(sequence), rows)
    reset_stream = "Dreyer2026Proteus/P01|1"
    reset_rows = p[data["stream"][p] == reset_stream][:64]
    if np.array_equal(sequence, rows) or len(reset_rows) != 64 or held[reset_rows].any():
        raise ValueError("within0 delivery is inactive")
    api = SimpleNamespace(ARCHS=ARCHS, make=lambda family, d, seed: make(ARCHS[family][0], d, seed))
    candidates = {family: saved_fit(api, family, person, data, rows, sequence, reset_rows, fit_sha)
                  for family in ARCHS}
    sinc = fresh_sinc(person, rows, sequence, deliveries, fit_sha)
    output = dict(person=person, order=ORDER, rows=len(rows), candidates=candidates, fresh_sinc=sinc,
                  held_rows_sha256=rows_sha(rows), session_count=len(streams), fit_rows_sha256=fit_sha,
                  held_labels_opened=False, prediction_mode="frozen",
                  reset_interleaving=f"{reset_stream}, 64 windows, discarded",
                  gpu=gpu, node=os.environ["SLURMD_NODENAME"], torch=torch.__version__,
                  p04_smoke_sha256=os.environ["NI26_WITHIN0_P04_SHA"],
                  delivery_sha256=rows_sha(p),
                  source_sha256=file_sha(HERE / "source.sha256"))
    encoded = json.dumps(output, indent=2, sort_keys=True, allow_nan=False)
    receipt = HERE / f"{person.lower()}.json"
    with receipt.open("x") as handle:
        handle.write(encoded + "\n")
    receipt.chmod(0o444)
    print(encoded, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=int, choices=(2, 3, 5, 6, 7, 8, 9, 10))
    main(PEOPLE[parser.parse_args().fold - 1])
