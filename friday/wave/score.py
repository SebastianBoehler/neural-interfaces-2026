"""Score held-excluded waveform checkpoints against the matched §430 cap250 replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


WORK = Path("/mnt/lustre/work/gehler/gfh527")
BASE = WORK / "ni26_s318_coral_fbl_20261117"
DATA = BASE / "data/proteus_windows.npz"
PERMS = BASE / "runs/shuffle_perms.npz"
DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
PERMS_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
STOCK_SCORE = 76.10391516713594
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
PEOPLE = tuple(f"P{i:02d}" for i in range(1, 11))
ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
HISTORY = ("genuine", "donor", "blank")
ARMS = ("stock", *(f"wave_{h}" for h in HISTORY), *(f"full_{h}" for h in HISTORY))
COUNTS = np.array([2, 2, 2, 6, 2, 2, 2, 2, 2, 6])


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def check_actor(path, kind, seed, person, actor_pin, people, full):
    proof = path.with_suffix(".json")
    record = json.loads(proof.read_text())
    expected = dict(kind=kind, seed=seed, held=person, actor=actor_pin, data=DATA_SHA, steps=600 if full else 4,
                    orders=PERMS_SHA, checkpoint=sha(path))
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
    import torch
    payload = torch.load(path, weights_only=True, map_location="cpu")
    check(payload["kind"] == kind and payload["held"] == person, f"wrong saved actor: {path}")
    return {"receipt": sha(proof), "checkpoint": expected["checkpoint"]}


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


def make_bank(kind, seed, people_to_score, meta, ages, baseline):
    from fit import read_inputs, score_checkpoint

    full = set(people_to_score) == set(PEOPLE)
    actor_pin = sha(Path("actor.sha256"))
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
        proof[person] = check_actor(path, kind, seed, person, actor_pin, people, full)
        values = score_checkpoint(path, x, people, streams, perms, device="cpu")
        rows = np.flatnonzero(people == person)
        for (order, arm), pred in checked_decisions(values, rows, ages, baseline).items():
            decisions[f"{person}|{order}|{arm}"] = pred
    if full:
        check(len(decisions) == len(PEOPLE) * len(ORDERS) * len(ARMS), "incomplete decisions")
    return decisions, proof


def committed(stem):
    decisions = Path(f"{stem}_decisions.npz")
    proof = json.loads(Path(f"{stem}_before_labels.json").read_text())
    check(sha(decisions) == proof["decisions"], f"changed decision bank: {stem}")
    with np.load(decisions, allow_pickle=False) as bank:
        return {k: bank[k] for k in bank}


def summarize(bank, pairs, meta, interval, metric, compare_history=False):
    with np.load(DATA, allow_pickle=False) as data:
        labels = data["y"]
    names = sorted({name for pair in pairs for name in pair})
    tables = {name: np.empty((len(PEOPLE), len(ORDERS))) for name in names}
    for p, person in enumerate(PEOPLE):
        rows = meta["person"] == person
        for i, order in enumerate(ORDERS):
            for name in names:
                pred = bank[f"{person}|{order}|{name}"]
                result = metric(labels[rows], pred, meta, rows)
                check(result["n_cells"] == COUNTS[p], f"unpaired metric cells {person}")
                tables[name][p, i] = 100 * result["balanced_accuracy_cell_mean"]
    values = {name: interval(table, COUNTS) for name, table in tables.items()}
    check(abs(values["stock"]["pp"] - STOCK_SCORE) < 1e-10, "stock metric mismatch")
    contrasts = {}
    for left, right in pairs:
        delta = tables[left] - tables[right]
        changed = sum(np.count_nonzero(bank[f"{p}|{o}|{left}"] != bank[f"{p}|{o}|{right}"])
                      for p in PEOPLE for o in ORDERS)
        contrasts[f"{left}-{right}"] = dict(
            **interval(delta, COUNTS), changed=int(changed),
            orders=dict(zip(ORDERS, np.average(delta, weights=COUNTS, axis=0).tolist())))
    if compare_history:
        delta = ((tables["full_genuine"] - tables["full_blank"])
                 - (tables["prototype_full_genuine"] - tables["prototype_full_blank"]))
        contrasts["centered-history_minus_prototype-history"] = dict(
            **interval(delta, COUNTS),
            orders=dict(zip(ORDERS, np.average(delta, weights=COUNTS, axis=0).tolist())))
    return {"absolute": values, "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "score", "compare"))
    parser.add_argument("kind", nargs="?", choices=("prototype", "centered"))
    parser.add_argument("seed", nargs="?", type=int, choices=(460, 1460))
    args = parser.parse_args()
    owner, interval, metric = upstream()
    meta, ages, baseline = stock_reference(owner)
    if args.mode == "preflight":
        print("PASS: pinned data, eight stock orders, exact per-row zero-change")
        return
    check(args.kind and args.seed, "kind and seed required")
    stem = f"{args.kind}_{args.seed}"
    if args.mode == "compare":
        check(args.kind == "centered", "compare requires the centered successor")
        prototype = committed(f"prototype_{args.seed}")
        centered = committed(stem)
        pairs = [("full_genuine", "stock"), ("full_genuine", "full_blank"),
                 ("prototype_full_genuine", "prototype_full_blank")]
        for key, decisions in prototype.items():
            person, order, arm = key.split("|")
            if arm == "stock":
                check(np.array_equal(decisions, centered[key]), f"unpaired stock: {key}")
            else:
                centered[f"{person}|{order}|prototype_{arm}"] = decisions
        pairs += [(f"{scope}_genuine", f"prototype_{scope}_genuine")
                  for scope in ("wave", "full")]
        result = summarize(centered, pairs, meta, interval, metric, compare_history=True)
        result["parent_decision_sha256"] = {
            arm: sha(Path(f"{arm}_{args.seed}_decisions.npz")) for arm in ("prototype", "centered")
        }
        Path(f"comparison_{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")
        return
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
    result = summarize(committed(stem), pairs, meta, interval, metric)
    Path(f"{stem}_result.json").write_text(json.dumps({**proof, **result}, indent=2) + "\n")
    print(json.dumps({arm: row["pp"] for arm, row in result["absolute"].items()}))


if __name__ == "__main__":
    main()
