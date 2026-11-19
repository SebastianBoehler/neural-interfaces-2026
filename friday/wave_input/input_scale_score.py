"""Score F11 causal sensor scaling against its matched training and input controls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch


WORK = Path("/mnt/lustre/work/gehler/gfh527")
F9_SCORER = WORK / "ni26_friday_wave_norm_20261118"
F9_PIN = "2e3032e19ed3cbb361138599ebac577c69d870c00b4b0a62e303f27b56a578ba"
KINDS = ("raw_scale_genuine", "raw_scale_donor")
MODELS = {KINDS[0]: "genuine_trained", KINDS[1]: "donor_trained"}
HISTORIES = ("genuine", "sensor_donor", "sensor_identity", "blank")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def frozen_scorer():
    manifest = F9_SCORER / "score.sha256"
    check(sha(manifest) == F9_PIN, "archived scoring source changed")
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        check(sha(F9_SCORER / relative) == expected, f"changed scoring source {relative}")
    sys.path.append(str(F9_SCORER))
    import friday_wave_score

    return friday_wave_score


def checked_actor(f9, path, kind, seed, person, actor, people, streams, smoke):
    base = f9.check_actor(path, kind, seed, person, actor, people, not smoke, {})
    record = json.loads(path.with_suffix(".json").read_text())
    payload = torch.load(path, weights_only=True, map_location="cpu")
    donor_name = min(set(streams[people != person]))
    donor_rows = np.flatnonzero(streams == donor_name)
    check(payload["donor_stream"] == donor_name and record["donor_stream"] == donor_name
          and record["donor_person"] == donor_name.split("|")[0]
          and record["donor_rows"] == digest(donor_rows, "<i8"), f"wrong F11 donor rows: {path}")
    guard = {
        "reference": ((41,), "reference_sha256"),
        "donor_power": ((len(donor_rows), 41), "donor_power_sha256"),
    }
    for key, (shape, receipt_key) in guard.items():
        values = payload[key]
        check(values.dtype == torch.float64 and tuple(values.shape) == shape
              and values.isfinite().all() and (values >= 0).all(), f"invalid {key}: {path}")
        check(record[receipt_key] == digest(values.numpy(), "<f8"), f"wrong {key} digest: {path}")
    check(bool((payload["reference"] > 0).all()), f"nonpositive scale reference: {path}")
    check(record["reference_rows"] == record["train_rows"] and record["prefix_windows"] == 20
          and np.isfinite(record["active_gain"]) and record["active_gain"] > 1e-4,
          f"wrong F11 fit rows or activity: {path}")
    shared_keys = ("reference_sha256", "donor_power_sha256", "reference_rows", "donor_rows",
                   "episode_rows", "donor_episode_rows", "initial_model")
    shared = {key: record[key] for key in shared_keys}
    check(all(re.fullmatch(r"[a-f0-9]{64}", value) for value in shared.values()),
          f"invalid matched protocol hashes: {path}")
    return {**base, **shared, "donor_stream": donor_name, "active_gain": record["active_gain"]}


def checked_logits(values, rows, ages, orders):
    expected_age = np.stack([ages[o][rows] for o in orders])
    check(np.array_equal(values["index"], rows) and np.array_equal(values["age"], expected_age),
          "held rows or pinned delivery ages differ")
    first = expected_age == 0
    for arm in HISTORIES:
        logits = values[arm]
        check(logits.shape == (*expected_age.shape, 3) and np.isfinite(logits).all(),
              f"invalid held logits: {arm}")
        np.testing.assert_array_equal(values["genuine"][first], logits[first])
    for arm in HISTORIES[1:]:
        check(np.any(np.abs(values["genuine"] - values[arm]) > 1e-5), f"inactive held intervention: {arm}")


def decide(values, name, order_index, stock):
    result = {}
    for arm in HISTORIES:
        logits = values[arm][order_index].astype(np.float64)
        centered = logits - logits.mean(axis=1, keepdims=True)
        result[f"wave_{name}_{arm}"] = logits.argmax(1)
        result[f"full_{name}_{arm}"] = (stock + .20 * np.clip(centered, -3, 3)).argmax(1)
    return {key: value.astype(np.uint8) for key, value in result.items()}


def paired_arms():
    pairs = []
    for scope in ("wave", "full"):
        genuine = f"{scope}_genuine_trained_genuine"
        donor = f"{scope}_donor_trained_genuine"
        pairs.extend((name, "stock") for name in (genuine, donor, f"{scope}_f7"))
        pairs.extend([(genuine, donor), (genuine, f"{scope}_f7")])
        for kind in KINDS:
            base = f"{scope}_{MODELS[kind]}"
            pairs.extend((f"{base}_genuine", f"{base}_{arm}") for arm in HISTORIES[1:])
    return pairs


def run(mode, seed):
    torch.set_num_threads(1)
    f9 = frozen_scorer()
    owner, interval, metric = f9.upstream()
    meta, ages, baseline = f9.stock_reference(owner)
    if mode == "preflight":
        print("PASS: pinned EEG, deliveries, and exact zero-change cap250")
        return

    from fit import read_inputs
    from input_scale_infer import score_checkpoint
    from style_score_inputs import f7_bank

    x, people, streams, interfaces, perms = read_inputs(f9.DATA, f9.PERMS)
    metadata_streams = np.char.add(np.char.add(people, "|"), meta["session"])
    check(np.array_equal(people, meta["person"]) and np.array_equal(interfaces, meta["interface"])
          and np.array_equal(streams, metadata_streams) and tuple(perms) == f9.ORDERS,
          "held raw data differs from stock metadata or order")
    smoke = mode == "smoke"
    actor = sha(Path("actor.sha256"))
    bank, proofs = {}, {}
    f7, f7_proof = ({}, None) if smoke else f7_bank(WORK, seed, f9.PEOPLE, f9.committed)
    for person in ("P01",) if smoke else f9.PEOPLE:
        rows = np.flatnonzero(people == person)
        matched = {}
        for kind in KINDS:
            prefix = "smoke_" if smoke else ""
            path = Path("checkpoints") / f"{prefix}{kind}_{seed}_{person}.pt"
            proof = checked_actor(f9, path, kind, seed, person, actor, people, streams, smoke)
            if matched:
                fields = ("reference_sha256", "donor_power_sha256", "reference_rows", "donor_rows",
                          "episode_rows", "donor_episode_rows", "initial_model")
                check(all(proof[k] == matched[k] for k in fields), "matched F11 training inputs differ")
            matched = proof
            proofs[f"{person}|{kind}"] = proof
            values = score_checkpoint(path, x, people, streams, perms, device="cpu")
            checked_logits(values, rows, ages, f9.ORDERS)
            for i, order in enumerate(f9.ORDERS):
                stock = baseline[order][rows]
                key = f"{person}|{order}"
                labels = stock.argmax(1).astype(np.uint8)
                np.testing.assert_array_equal((stock + np.zeros_like(stock)).argmax(1), labels)
                bank[f"{key}|stock"] = labels
                if not smoke:
                    np.testing.assert_array_equal(labels, f7[f"{key}|stock"])
                    for scope in ("wave", "full"):
                        bank[f"{key}|{scope}_f7"] = f7[f"{key}|{scope}_genuine"]
                bank.update({f"{key}|{name}": pred for name, pred in
                             decide(values, MODELS[kind], i, stock).items()})
    if smoke:
        print("PASS: matched P01 F11 source receipts, held score path, active intervention, ages and first windows")
        return
    path = Path(f"input_scale_{seed}_decisions.npz")
    np.savez_compressed(path, **bank)
    receipt = dict(seed=seed, decisions=sha(path), actor=actor, folds=proofs, f7=f7_proof,
                   stock=owner.REPORT_SHA, data=f9.DATA_SHA, orders=f9.PERMS_SHA,
                   scorer=sha(Path("score.sha256")), replay=sha(Path("recording-replay.log")))
    path.with_name(f"input_scale_{seed}_before_labels.json").write_text(json.dumps(receipt, indent=2) + "\n")
    check(sha(path) == receipt["decisions"], "blind decision bank changed")
    with np.load(path, allow_pickle=False) as frozen:
        sealed = {k: frozen[k] for k in frozen}
    scores = f9.summarize(sealed, paired_arms(), meta, interval, metric)
    path.with_name(f"input_scale_{seed}_result.json").write_text(json.dumps({**receipt, **scores}, indent=2) + "\n")
    print(json.dumps({key: scores["absolute"][key]["pp"] for key in
                      ("wave_genuine_trained_genuine", "wave_donor_trained_genuine", "wave_f7", "stock")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "score"))
    parser.add_argument("seed", nargs="?", type=int, choices=(460, 1460))
    args = parser.parse_args()
    check(args.mode == "preflight" or args.seed is not None, "expected an F11 seed")
    run(args.mode, args.seed)
