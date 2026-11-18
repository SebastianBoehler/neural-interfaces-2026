"""Score held-excluded waveform checkpoints against the matched §430 cap250 replay."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from score_metrics import ORDERS, PEOPLE, score_bank
from score_motor import MOTOR_KINDS, pretraining, sha
from style_score_inputs import STYLE_KINDS, f7_bank


WORK = Path("/mnt/lustre/work/gehler/gfh527")
BASE = WORK / "ni26_s318_coral_fbl_20261117"
DATA = BASE / "data/proteus_windows.npz"
PERMS = BASE / "runs/shuffle_perms.npz"
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
PERMS_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
NUMPY_STRING_ACTOR = "58e6e23425f0ee195a62916d130dd406c7a56b75aa77b3b35323846755af8fc2"
PARENTS = {
    "ni26_match456_v2_20261118/stocks456.py":
        "184522613cd35e54435fb3a5688949d9b9eb6cd046e18a0be361c40b3a86902c",
    "ni26_history443_20261118/score.py":
        "a1d56e39f4b92e1e06795fbc7d2a264243cd6dd6516e62f10132afa46e352f7f",
    "ni26_wave379_v1_20261118/score379.py":
        "5adfcb30ea9c88ae4f3bc2ef128b502dba5bfc16a05054bc4f57f894289dcbaa",
    "ni26_fusion383_v1_20261118/fusion_inputs.py":
        "0c49a4266c34841a7996e65591c0b6b89a8dd849868586c364159d5d195259cc",
}
HISTORY = ("genuine", "donor", "blank")
COMPARATORS = {
    "centered": ("prototype",),
    "chronological": ("symmetric",),
    "coherent_time": ("scattered_time", "chronological"),
    "scattered_time": ("chronological",),
    "motor_transfer": ("motor_null", "chronological"),
    "motor_null": ("chronological",),
    "raw_masked": ("raw_supervised", "chronological"),
    "raw_supervised": ("chronological",),
    "raw_style_full": ("raw_style_diagonal", "raw_supervised"),
}
RAW_KINDS = ("raw_supervised", "raw_masked", *STYLE_KINDS)
KINDS = tuple(sorted(set(COMPARATORS) | {r for arms in COMPARATORS.values() for r in arms}))
ARMS = ("stock", *(f"wave_{h}" for h in HISTORY), *(f"full_{h}" for h in HISTORY))


def check(ok, why):
    if not ok:
        raise ValueError(why)


def upstream():
    for name, digest in PARENTS.items():
        path = WORK / name
        check(sha(path) == digest, f"changed parent {name}")
        sys.path.append(str(path.parent))
    import fusion_inputs
    import score as history
    import stocks456

    check(stocks456.ref.ORDERS == ORDERS, "changed official order list")
    check(stocks456.ref.PEOPLE == PEOPLE, "changed held-person list")
    check(sha(DATA) == DATA_SHA and sha(PERMS) == PERMS_SHA, "changed raw data/order")
    return stocks456, history.interval, fusion_inputs.official()


def stock_reference(owner):
    meta, ages, _, stock = owner.inputs()
    result = {}
    for order in ORDERS:
        log_prob = owner.log_probability(stock[order], ages[order])
        labels = stock[order].argmax(1)
        check(np.array_equal((log_prob + np.zeros_like(log_prob)).argmax(1), labels),
              f"zero-change mismatch {order}")
        check(labels.shape == (2655,), f"unexpected stock rows {order}")
        result[order] = log_prob
    return meta, ages, result


def check_actor(path, kind, seed, person, actor_pin, people, full, lineage):
    proof = path.with_suffix(".json")
    record = json.loads(proof.read_text())
    expected = dict(kind=kind, seed=seed, held=person, actor=actor_pin, data=DATA_SHA, steps=600 if full else 4,
                    orders=PERMS_SHA, checkpoint=sha(path))
    if kind in STYLE_KINDS:
        expected["synthetic_updates"] = expected["steps"] // 2
    check(all(record.get(name) == value for name, value in expected.items()),
          f"checkpoint or training receipt mismatch: {path}")
    check(sorted(record["train_people"]) == sorted(set(PEOPLE) - {person}),
          f"held person entered training roster: {path}")
    train = np.flatnonzero(people != person).astype("<i8")
    check(record["train_rows"] == hashlib.sha256(train.tobytes()).hexdigest(),
          f"source row inventory mismatch: {path}")
    check(record["held_labels_opened"] is False, f"held labels were read: {path}")
    check(record["device"] == "cuda" and "H100" in record["gpu"],
          f"unexpected training hardware: {path}")
    if lineage:
        check(all(record.get(k) == lineage[k] for k in ("pretrain", "pretrain_source")),
              f"wrong pretraining initialization: {path}")
    import torch
    payload = torch.load(path, weights_only=True, map_location="cpu")
    check(payload["kind"] == kind and payload["held"] == person, f"wrong saved actor: {path}")
    provenance = {"synthetic_updates": expected["synthetic_updates"]} if kind in STYLE_KINDS else {}
    return {"receipt": sha(proof), "checkpoint": expected["checkpoint"], **lineage, **provenance}


def checked_decisions(values, rows, ages, baseline):
    check(np.array_equal(values["index"], rows), "held rows differ")
    expected_age = np.stack([ages[o][rows] for o in ORDERS])
    check(np.array_equal(values["age"], expected_age), "session delivery ages differ")
    first = expected_age == 0
    for arm in HISTORY:
        logits = values[arm]
        check(logits.shape == (*expected_age.shape, 3) and np.isfinite(logits).all(),
              f"invalid logits {arm}")
        check(np.array_equal(logits[first], values["blank"][first]),
              f"first window used session history: {arm}")
    check(np.any(np.abs(values["genuine"] - values["blank"]) > 1e-5),
          "genuine waveform history had no effect")
    check(np.any(np.abs(values["genuine"] - values["donor"]) > 1e-5),
          "donor waveform history had no effect")
    decisions = {}
    for i, order in enumerate(ORDERS):
        stock = baseline[order][rows]
        decisions[order, "stock"] = stock.argmax(1)
        for arm in HISTORY:
            logits = values[arm][i].astype(np.float64)
            correction = 0.20 * np.clip(logits - logits.mean(1, keepdims=True), -3, 3)
            decisions[order, f"wave_{arm}"] = logits.argmax(1)
            decisions[order, f"full_{arm}"] = (stock + correction).argmax(1)
    return {key: value.astype(np.uint8) for key, value in decisions.items()}


def checkpoint_loader(actor_pin):
    """Keep the weights-only loader for the pinned F1/F2 NumPy donor string."""
    if actor_pin != NUMPY_STRING_ACTOR:
        return nullcontext()
    from numpy._core.multiarray import scalar
    from torch.serialization import safe_globals

    return safe_globals([scalar, np.dtype, np.dtypes.StrDType])


def make_bank(kind, seed, people_to_score, meta, ages, baseline):
    from fit import read_inputs, score_checkpoint

    full = set(people_to_score) == set(PEOPLE)
    actor_pin = sha(Path("actor.sha256"))
    lineage = pretraining(kind, seed) if kind in MOTOR_KINDS else {}
    x, people, streams, interfaces, perms = read_inputs(DATA, PERMS)
    check(np.array_equal(people, meta["person"]), "EEG row owners differ")
    expected_streams = np.char.add(np.char.add(people, "|"), meta["session"])
    check(np.array_equal(streams, expected_streams), "EEG session row owners differ")
    check(np.array_equal(interfaces, meta["interface"]), "EEG interface rows differ")
    check(tuple(perms) == ORDERS, "EEG order names differ")
    decisions, proof = {}, {}
    for person in people_to_score:
        prefix = "" if full else "smoke_"
        path = Path("checkpoints") / f"{prefix}{kind}_{seed}_{person}.pt"
        with checkpoint_loader(actor_pin):
            proof[person] = check_actor(path, kind, seed, person, actor_pin, people, full, lineage)
            values = score_checkpoint(path, x, people, streams, perms, device="cpu")
        rows = np.flatnonzero(people == person)
        for (order, arm), pred in checked_decisions(values, rows, ages, baseline).items():
            decisions[f"{person}|{order}|{arm}"] = pred
    if full:
        check(len(decisions) == len(PEOPLE) * len(ORDERS) * len(ARMS), "incomplete decisions")
    return decisions, proof


def committed(stem, root=Path("."), proof_sha=None):
    decisions = root / f"{stem}_decisions.npz"
    proof_path = root / f"{stem}_before_labels.json"
    if proof_sha is not None:
        check(sha(proof_path) == proof_sha, f"changed parent proof: {stem}")
    proof = json.loads(proof_path.read_text())
    check(proof["data"] == DATA_SHA and proof["orders"] == PERMS_SHA,
          f"different row or delivery inventory: {stem}")
    check(sha(decisions) == proof["decisions"], f"changed decision bank: {stem}")
    with np.load(decisions, allow_pickle=False) as bank:
        check(set(bank.files) == {f"{p}|{o}|{arm}" for p in PEOPLE for o in ORDERS for arm in ARMS},
              f"incomplete parent decisions: {stem}")
        return {k: bank[k] for k in bank}, proof["decisions"]


def summarize(bank, pairs, meta, interval, metric, ref_kinds=(), evaluate_motor=False, ages=None,
              split_first=False):
    with np.load(DATA, allow_pickle=False) as data:
        labels = data["y"]
    return score_bank(bank, pairs, meta, interval, metric, labels, ref_kinds, evaluate_motor, ages,
                      split_first)


def comparator(kind, ref, seed):
    if kind in STYLE_KINDS and ref == "raw_supervised":
        return f7_bank(WORK, seed, PEOPLE, committed)
    if kind not in (*MOTOR_KINDS, *RAW_KINDS) or ref != "chronological":
        return committed(f"{ref}_{seed}")
    proof = {
        460: "86561c6d4f93ff77974f21f3130d2824f71c4ea92952ab287a4efb07e31200c2",
        1460: "fd8238f50dcac10e89e7153938ae1c6f18b7496f66f1c3e8d79c1ef763c67163",
    }
    return committed(f"chronological_{seed}", WORK / "ni26_friday_wave_time_20261118", proof[seed])


def compare(kind, seed, meta, ages, interval, metric, reference=None):
    check(kind in COMPARATORS, f"no paired comparator for {kind}")
    ref_kinds = COMPARATORS[kind]
    if reference is not None:
        check(reference in ref_kinds, f"unregistered comparator: {reference}")
        ref_kinds = (reference,)
    candidate, digest = committed(f"{kind}_{seed}")
    proofs = {kind: digest}
    pairs = [("full_genuine", "stock")]
    for scope in ("wave", "full"):
        pairs.append((f"{scope}_genuine", f"{scope}_blank"))
    for ref in ref_kinds:
        bank, proofs[ref] = comparator(kind, ref, seed)
        for key, decisions in bank.items():
            person, order, arm = key.split("|")
            if arm == "stock":
                check(np.array_equal(decisions, candidate[key]), f"unpaired stock: {key}")
            else:
                candidate[f"{person}|{order}|{ref}_{arm}"] = decisions
        for scope in ("wave", "full"):
            genuine, blank = f"{scope}_genuine", f"{scope}_blank"
            pairs.extend([(f"{ref}_{genuine}", f"{ref}_{blank}"),
                          (genuine, f"{ref}_{genuine}")])
    result = summarize(candidate, pairs, meta, interval, metric, ref_kinds,
                       evaluate_motor=kind in MOTOR_KINDS, ages=ages if kind in RAW_KINDS else None,
                       split_first=kind in STYLE_KINDS)
    result["parent_decision_sha256"] = proofs
    suffix = f"_vs_{reference}" if reference else ""
    Path(f"comparison_{kind}_{seed}{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "score", "compare"))
    parser.add_argument("kind", nargs="?", choices=KINDS)
    parser.add_argument("seed", nargs="?", type=int, choices=(460, 1460))
    parser.add_argument("reference", nargs="?")
    args = parser.parse_args()
    owner, interval, metric = upstream()
    meta, ages, baseline = stock_reference(owner)
    if args.mode == "preflight":
        print("PASS: pinned data, eight stock orders, exact per-row zero-change")
        return
    check(args.kind and args.seed, "kind and seed required")
    stem = f"{args.kind}_{args.seed}"
    if args.mode == "compare":
        compare(args.kind, args.seed, meta, ages, interval, metric, args.reference)
        return
    check(args.reference is None, "reference applies only to compare")
    people = PEOPLE[:1] if args.mode == "smoke" else PEOPLE
    bank, actors = make_bank(args.kind, args.seed, people, meta, ages, baseline)
    if args.mode == "smoke":
        print(f"PASS: {stem} held P01 checkpoint, age/reset, and active genuine/donor checks")
        return
    decision_path = Path(f"{stem}_decisions.npz")
    np.savez_compressed(decision_path, **bank)
    proof = dict(decisions=sha(decision_path), actors=actors, data=DATA_SHA,
                 orders=PERMS_SHA, stock=owner.REPORT_SHA, source=sha(Path("score.sha256")),
                 replay=sha(Path("recording-replay.log")))
    Path(f"{stem}_before_labels.json").write_text(json.dumps(proof, indent=2) + "\n")
    pairs = [(f"{scope}_genuine", f"{scope}_{arm}")
             for scope in ("wave", "full") for arm in ("blank", "donor")]
    pairs += [(arm, "stock") for arm in ARMS if arm != "stock"]
    saved, _ = committed(stem)
    result = summarize(saved, pairs, meta, interval, metric,
                       evaluate_motor=args.kind in MOTOR_KINDS, ages=ages if args.kind in RAW_KINDS else None,
                       split_first=args.kind in STYLE_KINDS)
    Path(f"{stem}_result.json").write_text(json.dumps({**proof, **result}, indent=2) + "\n")
    print(json.dumps({arm: row["pp"] for arm, row in result["absolute"].items()}))


if __name__ == "__main__":
    main()
