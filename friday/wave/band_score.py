"""Score the matched F12 frontend initializations in isolated held-person replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from band_frontend import KINDS, BandRawDecoder, initial_receipt, tensor_bytes


WORK = Path("/mnt/lustre/work/gehler/gfh527")
F9 = WORK / "ni26_friday_wave_norm_20261118"
F9_SCORER = "2e3032e19ed3cbb361138599ebac577c69d870c00b4b0a62e303f27b56a578ba"
F9_PROOFS = {
    460: "2b39075ccacf7d18e7c930cd31d4caf53337d56fc47f9990a34bb48e5b7ca1d2",
    1460: "2f74dcb7d92c52c80ba372e70cbdd3ac8ada6ca1bdeeb708118558b4330c731c",
}
ANCHOR = WORK / "ni26_s323_normsplit_20261117/whole_results.json"
ANCHOR_SHA = "89dc9aa77860b8ba4f06925294565d50eb7cf5874f310d598bdad24b4d04a92d"
NAMES = {KINDS[0]: "spectral", KINDS[1]: "random"}
ARMS = ("genuine", "donor", "blank")
PAIRS = (("spectral_genuine", "random_genuine"), ("spectral_genuine", "f7"),
         ("spectral_genuine", "f9"), ("spectral_genuine", "stock"),
         ("random_genuine", "f7"), ("random_genuine", "f9"),
         ("spectral_genuine", "spectral_donor"), ("spectral_genuine", "spectral_blank"),
         ("random_genuine", "random_donor"), ("random_genuine", "random_blank"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def controls():
    manifest = F9 / "score.sha256"
    check(sha(manifest) == F9_SCORER, "changed scorer manifest")
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        check(sha(F9 / name) == digest, f"changed scorer source: {name}")
    check(sha(ANCHOR) == ANCHOR_SHA, "changed historical S1C result")
    sys.path.append(str(F9))
    import friday_wave_score
    return friday_wave_score


def checked_actor(ref, path, kind, seed, person, actor, people, streams, initial, smoke, gpu_family):
    proof_path = path.with_suffix(".json")
    receipt = json.loads(proof_path.read_text())
    expected = dict(kind=kind, seed=seed, held=person, actor=actor, data=ref.DATA_SHA,
                    orders=ref.PERMS_SHA, checkpoint=sha(path), steps=4 if smoke else 600,
                    device="cuda", held_labels_opened=False)
    check(all(receipt.get(key) == value for key, value in expected.items())
          and gpu_family in receipt["gpu"], f"wrong actor source or GPU family: {path}")
    source_rows = np.flatnonzero(people != person).astype("<i8")
    check(sorted(receipt["train_people"]) == sorted(set(ref.PEOPLE) - {person})
          and receipt["train_rows"] == hashlib.sha256(source_rows.tobytes()).hexdigest(),
          f"held-person source roster or row identity changed: {path}")
    if gpu_family == "A100":
        check(receipt.get("cudnn_tf32") is False, f"A100 convolution was not full FP32: {path}")
    standard = {"receipt": sha(proof_path), "checkpoint": expected["checkpoint"],
                "gpu": receipt["gpu"], "torch": receipt["torch"],
                "cudnn_tf32": receipt.get("cudnn_tf32")}
    for key in ("initial_fir", "initial_other", "parameters"):
        check(receipt[key] == initial[key], f"wrong independently recreated initialization: {key}/{path}")
    np.testing.assert_allclose(receipt["fir_norm"], initial["fir_norm"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(receipt["in_band"], initial["in_band"], atol=1e-6, rtol=0)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    donor = min(set(streams[people != person]))
    check(payload["kind"] == kind and payload["held"] == person and payload["donor_stream"] == donor,
          f"wrong actor kind, fold, or donor: {path}")
    fitted = payload["model"]["stem.temporal.weight"]
    digest = hashlib.sha256(tensor_bytes(fitted)).hexdigest()
    check(receipt["fitted_fir"] == digest and receipt["initial_fir"] != digest,
          f"inactive or mismatched FIR update: {path}")
    check(len(receipt["episode_rows"]) == 64, f"missing episode schedule pin: {path}")
    return {**standard, **{k: receipt[k] for k in ("episode_rows", "parameters", "initial_fir",
                                                  "initial_other", "fitted_fir", "fir_norm", "in_band")}}


def decisions(ref, values, rows, ages):
    expected_age = np.stack([ages[o][rows] for o in ref.ORDERS])
    check(np.array_equal(values["index"], rows) and np.array_equal(values["age"], expected_age),
          "held row or canonical delivery age mismatch")
    first = expected_age == 0
    for arm in ARMS:
        logits = values[arm]
        check(logits.shape == (*expected_age.shape, 3) and np.isfinite(logits).all(), f"invalid {arm} logits")
        np.testing.assert_array_equal(logits[first], values["blank"][first])
    for arm in ("donor", "blank"):
        check(np.max(np.abs(values["genuine"] - values[arm])) > 1e-5, f"inactive history control {arm}")
    return {arm: values[arm].argmax(-1).astype(np.uint8) for arm in ARMS}


def run(mode, seed, gpu_family):
    torch.set_num_threads(1)
    f9 = controls()
    owner, interval, metric = f9.upstream()
    meta, ages, baseline = f9.stock_reference(owner)
    if mode == "preflight":
        print("PASS: authenticated archive, ten people, eight permutations and exact stock zero")
        return

    from fit import read_inputs, score_checkpoint
    from style_score_inputs import f7_bank

    x, people, streams, interfaces, perms = read_inputs(f9.DATA, f9.PERMS)
    expected_streams = np.char.add(np.char.add(people, "|"), meta["session"])
    check(tuple(perms) == f9.ORDERS and np.array_equal(people, meta["person"])
          and np.array_equal(streams, expected_streams) and np.array_equal(interfaces, meta["interface"]),
          "derived held data differ from history")
    smoke = mode == "smoke"
    actor = sha(Path("actor.sha256"))
    f7, f7_proof = ({}, None) if smoke else f7_bank(WORK, seed, f9.PEOPLE, f9.committed)
    fixed, f9_proof = ({}, None) if smoke else f9.committed(f"raw_norm_fixed_{seed}", F9, F9_PROOFS[seed])
    expected = {}
    for kind in KINDS:
        torch.manual_seed(seed)
        expected[kind] = initial_receipt(BandRawDecoder(kind), kind)
    a, b = (expected[k] for k in KINDS)
    check(a["initial_other"] == b["initial_other"] and a["parameters"] == b["parameters"]
          and a["initial_fir"] != b["initial_fir"], "matched initialization changed")
    np.testing.assert_allclose(a["fir_norm"], b["fir_norm"], atol=1e-6, rtol=0)
    bank, proofs = {}, {}
    for person in ("P01",) if smoke else f9.PEOPLE:
        rows = np.flatnonzero(people == person)
        fold = {}
        for kind in KINDS:
            name = NAMES[kind]
            prefix = "smoke_" if smoke else ""
            path = Path("checkpoints") / f"{prefix}{kind}_{seed}_{person}.pt"
            fold[name] = checked_actor(f9, path, kind, seed, person, actor, people, streams,
                                       expected[kind], smoke, gpu_family)
            held = score_checkpoint(path, x, people, streams, perms, device="cpu")
            for arm, answer in decisions(f9, held, rows, ages).items():
                for i, order in enumerate(f9.ORDERS):
                    bank[f"{person}|{order}|{name}_{arm}"] = answer[i]
        check(fold["spectral"]["episode_rows"] == fold["random"]["episode_rows"], "unpaired training rows")
        proofs[person] = fold
        for order in f9.ORDERS:
            key = f"{person}|{order}"
            stock = baseline[order][rows]
            labels = stock.argmax(1).astype(np.uint8)
            np.testing.assert_array_equal((stock + np.zeros_like(stock)).argmax(1), labels)
            bank[f"{key}|stock"] = labels
            if not smoke:
                np.testing.assert_array_equal(labels, f7[f"{key}|stock"])
                np.testing.assert_array_equal(labels, fixed[f"{key}|stock"])
                bank[f"{key}|f7"] = f7[f"{key}|wave_genuine"]
                bank[f"{key}|f9"] = fixed[f"{key}|wave_genuine"]
    check(len({entry["torch"] for fold in proofs.values() for entry in fold.values()}) == 1,
          "paired source actors used different PyTorch releases")
    if smoke:
        print("PASS: paired P01 initialization, fitted FIRs, source episodes, causal replay, ages and first windows")
        return
    path = Path(f"band_{seed}_decisions.npz")
    np.savez_compressed(path, **bank)
    receipt = dict(seed=seed, actor=actor, decisions=sha(path), source=sha(Path("score.sha256")),
                   folds=proofs, stock=owner.REPORT_SHA, f7=f7_proof, f9=f9_proof,
                   data=f9.DATA_SHA, orders=f9.PERMS_SHA, replay=sha(Path("recording-replay.log")),
                   source_gpu=gpu_family, historical_f7_f9_gpu="H100")
    Path(f"band_{seed}_before_labels.json").write_text(json.dumps(receipt, indent=2) + "\n")
    check(sha(path) == receipt["decisions"], "blind decision bank changed")
    with np.load(path, allow_pickle=False) as saved:
        metrics = f9.summarize({k: saved[k] for k in saved}, PAIRS, meta, interval, metric)
    metrics["historical_s1c"] = {"score": 59.04527454907313, "sha256": ANCHOR_SHA,
                                "spectral_gap_pp": metrics["absolute"]["spectral_genuine"]["pp"] - 59.04527454907313}
    Path(f"band_{seed}_result.json").write_text(json.dumps({**receipt, **metrics}, indent=2) + "\n")
    print(json.dumps({k: metrics["absolute"][k]["pp"] for k in ("spectral_genuine", "random_genuine", "f7", "f9")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "score"))
    parser.add_argument("seed", nargs="?", choices=(460, 1460), type=int)
    parser.add_argument("--gpu-family", choices=("H100", "A100"), default="H100")
    args = parser.parse_args()
    check(args.mode == "preflight" or args.seed is not None, "expected a seed")
    run(args.mode, args.seed, args.gpu_family)
