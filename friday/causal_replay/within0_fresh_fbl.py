"""Seal the same fit-excluded fresh FBLight states on within0 delivery."""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from fresh_sinc import evaluate, replay, state_hash
from member_inputs import ORDERS, pinned_delivery
from refit_smoke import HERE, ROOT, TOL, changed, file_sha, waveform_data
from restore_refit import RELEASE, RELEASE_SHA, restore
from within0_frozen_probe import PEOPLE, old_base, rows_sha


ORDER = "within0"
NAME = "fbl"
QUAL = HERE.parent / "within0_frozen_probe_v4/qualification.json"
QUAL_SHA = "43fe6dc888a0cef5fd423976577785b671474be2cfa13668059bbb53839a44eb"
ARCHS = {NAME: ("fblight_ea_rot05_sbn", 5.)}


def inputs(order=ORDER):
    if file_sha(RELEASE / "source.sha256") != RELEASE_SHA or file_sha(QUAL) != QUAL_SHA:
        raise ValueError("fresh FBLight and original frozen qualification differ")
    archive = json.loads(QUAL.read_text())
    delivery = pinned_delivery(ROOT / "runs/shuffle_perms.npz")
    if archive["delivery_sha256"] != rows_sha(delivery[ORDERS.index(ORDER)]) or not archive["families"][NAME]["eligible"] or archive["held_labels_opened"]:
        raise ValueError("historical frozen conditioning differs")
    producer = json.loads((RELEASE / "members/fbl_s0.json").read_text())
    if (producer["held_labels_opened"] or producer["order"] != "full0" or producer["member"] != NAME
            or set(producer["fit_states"]) != set(PEOPLE)):
        raise ValueError("fresh source state provenance differs")
    return delivery[ORDERS.index(order)], producer["fit_states"]


def evaluate_fold(person, device="rtx", order=ORDER):
    from session_bn import make

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
    actual = "rtx" if "RTX 2080 Ti" in gpu else "a100" if "A100" in gpu else "other" if gpu else "cpu"
    if actual != device or torch.__version__ != "2.14.1+cu126":
        raise ValueError("frozen FBLight device or runtime differs")
    if device == "a100":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    if os.environ["SLURM_CPUS_PER_TASK"] != "8":
        raise ValueError("expected eight source threads")
    torch.set_num_threads(8)
    p, states = inputs(order)
    d = waveform_data()
    api = SimpleNamespace(ARCHS=ARCHS, make=lambda family, data, seed: make(ARCHS[family][0], data, seed))
    spec = states[person]
    model, fit, held = restore(api, d, NAME, person, RELEASE / "checkpoints" / f"fbl_s0_{person}.pt", spec["sha256"])
    if rows_sha(fit) != spec["fit_rows_sha"] or len(fit) != spec["n_fit"]:
        raise ValueError("fresh FBLight source excludes the wrong rows")
    rows, seq = np.flatnonzero(held), p[held[p]]
    np.testing.assert_array_equal(np.sort(seq), rows)
    if np.array_equal(seq, rows):
        raise ValueError("frozen replay did not see the actual shuffled delivery")
    state = state_hash(model.net)
    partner = "P02" if person == "P01" else "P01"
    distract = p[d["stream"][p] == f"Dreyer2026Proteus/{partner}|1"][:64]
    if len(distract) != 64 or held[distract].any():
        raise ValueError("replay interleave has the wrong source rows")
    for stream in np.unique(d["stream"][rows]):
        r = seq[d["stream"][seq] == stream]
        np.testing.assert_array_equal(model._aligned(r, True)[:1], model._aligned(r[:1], True))
        first = evaluate(model, r[:64])
        evaluate(model, distract)
        np.testing.assert_array_equal(first, evaluate(model, r[:64]))
    result, prefix = replay(model, d, rows, seq)
    if state_hash(model.net) != state:
        raise ValueError("frozen replay mutated the state")
    legacy = changed(result, old_base(NAME, rows, order))
    if legacy["changed"] or legacy["max_abs"] > TOL:
        raise ValueError(f"fresh frozen state no longer matches the qualified within0 reference: {legacy}")
    return rows, seq, result, prefix, legacy, spec


def replay_fold(person, smoke, order=ORDER):
    rows, seq, result, prefix, legacy, spec = evaluate_fold(person, order=order)
    path = HERE / "folds" / f"{person}.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, index=rows, delivery=seq, proba=result)
    path.chmod(0o444)
    receipt = dict(person=person, order=order, rows=len(rows), smoke=smoke, prefix_max_abs=prefix, legacy=legacy,
                   source=file_sha(HERE / "source.sha256"), checkpoint_sha256=spec["sha256"],
                   fit_rows_sha256=spec["fit_rows_sha"], held_labels_opened=False, exact_reset=True,
                   output=file_sha(path), euclidean_first_exact=True)
    out = path.with_suffix(".json")
    with out.open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    out.chmod(0o444)
    print("PASS fresh FBLight", person, file_sha(out), flush=True)


def scope(person):
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as z:
        fits = z["subject"] != f"Dreyer2026Proteus/{person}"
    return np.flatnonzero(~fits), rows_sha(np.flatnonzero(fits))


def load_fold(person, p, state, *, release=HERE, order=ORDER):
    path = release / "folds" / f"{person}.npz"
    proof = path.with_suffix(".json")
    r = json.loads(proof.read_text())
    rows, fit_sha = scope(person)
    if ((r["person"], r["order"], r["rows"], r["smoke"]) != (person, order, len(rows), person == "P01")
            or r["source"] != file_sha(release / "source.sha256") or r["held_labels_opened"]
            or not r["exact_reset"] or not r["euclidean_first_exact"]
            or r["checkpoint_sha256"] != state["sha256"] or r["fit_rows_sha256"] != fit_sha
            or fit_sha != state["fit_rows_sha"] or not 0 <= r["prefix_max_abs"] <= TOL
            or r["legacy"]["changed"] or not 0 <= r["legacy"]["max_abs"] <= TOL
            or r["output"] != file_sha(path) or any(f.stat().st_mode & 0o222 for f in (proof, path))):
        raise ValueError(f"unqualified authenticated fresh FBLight fold {person}")
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"], rows)
        np.testing.assert_array_equal(z["delivery"], p[np.isin(p, rows)])
        values = z["proba"]
    return rows, values, dict(receipt=file_sha(proof), checkpoint_sha256=state["sha256"], fit_rows_sha256=fit_sha)


def collect(order=ORDER):
    p, states = inputs(order)
    values, folds = np.full((len(p), 3), np.nan), {}
    for person, spec in states.items():
        rows, vals, folds[person] = load_fold(person, p, spec, order=order)
        values[rows] = vals
    effect = changed(values, old_base(NAME, np.arange(len(p)), order))
    if effect["max_abs"] <= 1e-8 or not np.allclose(values.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError("common fresh FBLight stream is inactive or incomplete")
    path = HERE / "frozen.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, index=np.arange(len(p)), delivery=p, proba=values)
    path.chmod(0o444)
    receipt = dict(source=file_sha(HERE / "source.sha256"), order=order, rows=len(p), bank=file_sha(path),
                   held_labels_opened=False, legacy_warm_parity=False, legacy_frozen_comparison=effect, folds=folds)
    out = HERE / "frozen.json"
    with out.open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    out.chmod(0o444)
    print("PASS complete shared FBLight bank", file_sha(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fold", "check", "collect"))
    parser.add_argument("id", type=int, nargs="?", choices=range(1, 11))
    parser.add_argument("--order", choices=(ORDER, "runs0"), default=ORDER)
    args = parser.parse_args()
    if args.mode == "fold":
        replay_fold(PEOPLE[args.id - 1], args.id == 1, args.order)
    elif args.mode == "check":
        delivery, states = inputs(args.order)
        print(load_fold("P01", delivery, states["P01"], order=args.order)[2]["receipt"], flush=True)
    else:
        collect(args.order)
