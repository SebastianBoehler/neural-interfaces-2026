"""Insert only the authenticated F9 fixed normalization update into §430 cap250."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


WORK = Path("/mnt/lustre/work/gehler/gfh527")
F9 = WORK / "ni26_friday_wave_norm_20261118"
PARENTS = {
    "actor.sha256": "4d601b39e4ab65dd519e1257b24829cf7ef3883c981f69834623fcc1cc7ed526",
    "score.sha256": "2e3032e19ed3cbb361138599ebac577c69d870c00b4b0a62e303f27b56a578ba",
}
BEFORE_LABELS = {
    460: "2b39075ccacf7d18e7c930cd31d4caf53337d56fc47f9990a34bb48e5b7ca1d2",
    1460: "2f74dcb7d92c52c80ba372e70cbdd3ac8ada6ca1bdeeb708118558b4330c731c",
}
KIND = "raw_norm_fixed"
HISTORY = ("genuine", "donor")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def verify_f9():
    for name, expected in PARENTS.items():
        manifest = F9 / name
        check(sha(manifest) == expected, f"F9 {name} changed")
        for line in manifest.read_text().splitlines():
            digest, relative = line.split(maxsplit=1)
            check(sha(F9 / relative) == digest, f"F9 source changed: {relative}")
    sys.path.insert(0, str(F9))


def f9_inputs(f9, meta, ages):
    from fit import read_inputs

    x, people, streams, interfaces, perms = read_inputs(f9.DATA, f9.PERMS)
    canonical_streams = np.char.add(np.char.add(people, "|"), meta["session"])
    check(np.array_equal(people, meta["person"]) and np.array_equal(streams, canonical_streams)
          and np.array_equal(interfaces, meta["interface"]), "F9 row metadata changed")
    for order, perm in perms.items():
        for name in sorted(set(streams)):
            rows = perm[streams[perm] == name]
            check(np.array_equal(ages[order][rows], np.arange(len(rows))), f"invalid source order {order}/{name}")
    return x, people, streams, perms


def readout_delta(model, z, logits, writes):
    from normalize import NormalizedRawDecoder
    from raw import RawDecoder
    from waveform import WIDTH

    prior = torch.cat([torch.zeros_like(writes[:1]), writes[:-1].cumsum(dim=0)])
    age = torch.arange(len(writes))
    normalized = NormalizedRawDecoder.read(model, z, logits, prior, age)
    original = RawDecoder.read(model, z, logits, prior[:, :-WIDTH], age)
    delta = normalized - original

    # The explicit feature calculation checks that prototype and style terms cancel.
    mass = 20 + age[:, None]
    mean = prior[:, -2*WIDTH:-WIDTH] / mass
    variance = (prior[:, -WIDTH:] + 20) / mass - mean.square()
    feature = ((z - mean) * variance.clamp_min(.25).rsqrt() - z).clamp(-2, 2)
    torch.testing.assert_close(delta, feature @ model.static.weight.T, atol=3e-6, rtol=1e-6)
    check(not delta[0].count_nonzero() and delta.isfinite().all(), "invalid first-window normalization")
    return normalized.numpy(), delta.numpy()


@torch.no_grad()
def isolate(path, inputs, reference):
    from normalize import NormalizedRawDecoder

    x, people, streams, perms = inputs
    payload = torch.load(path, weights_only=True, map_location="cpu")
    person, donor_stream = payload["held"], payload["donor_stream"]
    donor_name = min(set(streams[people != person]))
    check(donor_stream == donor_name, "F9 matched donor changed")
    model = NormalizedRawDecoder(KIND).eval()
    model.load_state_dict(payload["model"])
    check(torch.equal(model.prior(), torch.full_like(model.prior(), 20)), "fixed prior changed")
    donor_x = torch.from_numpy(x[streams == donor_stream])
    donor_z = model.encode(model.prepare(donor_x))
    donor = model.writes(donor_z, model.static(donor_z))
    rows = reference["index"]
    isolated = {arm: np.full(reference[arm].shape, np.nan, dtype=np.float64) for arm in HISTORY}
    for oi, perm in enumerate(perms.values()):
        for stream in sorted(set(streams[rows])):
            ordered = perm[streams[perm] == stream]
            pos = np.searchsorted(rows, ordered)
            z = model.encode(model.prepare(torch.from_numpy(x[ordered])))
            logits = model.static(z)
            np.testing.assert_array_equal(logits.numpy(), reference["blank"][oi, pos])
            current = {"genuine": model.writes(z, logits),
                       "donor": donor[torch.arange(len(z)) % len(donor)]}
            for arm, writes in current.items():
                normalized, delta = readout_delta(model, z, logits, writes)
                np.testing.assert_array_equal(normalized, reference[arm][oi, pos])
                delta = delta.astype(np.float64)
                isolated[arm][oi, pos] = .10 * np.clip(delta - delta.mean(axis=1, keepdims=True), -1, 1)
    for arm in HISTORY:
        values = isolated[arm]
        check(np.isfinite(values).all() and not values[reference["age"] == 0].any()
              and bool(values.any()), f"invalid isolated {arm} values")
    check(not np.array_equal(isolated["genuine"], isolated["donor"]), "donor intervention inactive")
    return isolated


def run(seed, smoke):
    torch.set_num_threads(1)
    verify_f9()
    import friday_wave_score as f9
    from fit import score_checkpoint

    owner, interval, metric = f9.upstream()
    meta, ages, baseline = f9.stock_reference(owner)
    inputs = f9_inputs(f9, meta, ages)
    stem = f"{KIND}_{seed}"
    archived, decision_sha = f9.committed(stem, F9, BEFORE_LABELS[seed])
    proof = json.loads((F9 / f"{stem}_before_labels.json").read_text())
    check(proof["source"] == PARENTS["score.sha256"] and proof["stock"] == owner.REPORT_SHA,
          "F9 scorer or stock proof changed")
    people = ("P04",) if smoke else f9.PEOPLE
    bank, actors, active = {}, {}, {}
    x, row_people, streams, perms = inputs
    for person in people:
        path = F9 / "checkpoints" / f"{stem}_{person}.pt"
        actors[person] = f9.check_actor(path, KIND, seed, person, PARENTS["actor.sha256"], row_people, True, {})
        check(actors[person] == proof["actors"][person], f"wrong F9 fold weights {person}")
        reference = score_checkpoint(path, x, row_people, streams, perms, device="cpu")
        rows = np.flatnonzero(row_people == person)
        original = f9.checked_decisions(reference, rows, ages, baseline)
        for (order, arm), predictions in original.items():
            key = f"{person}|{order}|{arm}"
            check(np.array_equal(predictions, archived[key]), f"F9 archived decisions differ: {key}")
        corrections = isolate(path, inputs, reference)
        active[person] = {arm: int(np.count_nonzero(v)) for arm, v in corrections.items()}
        for oi, order in enumerate(f9.ORDERS):
            stock = baseline[order][rows]
            unchanged = original[order, "stock"]
            np.testing.assert_array_equal((stock + np.zeros_like(stock)).argmax(1), unchanged)
            bank[f"{person}|{order}|stock"] = unchanged
            bank[f"{person}|{order}|f9_full"] = original[order, "full_genuine"]
            for arm, values in corrections.items():
                pred = (stock + values[oi]).argmax(1).astype(np.uint8)
                first = ages[order][rows] == 0
                np.testing.assert_array_equal(pred[first], unchanged[first])
                bank[f"{person}|{order}|{arm}"] = pred
    if smoke:
        print("PASS: P04 exact F9 logits and archived decisions, age, isolated active reads, and stock first windows")
        return
    bank_path = Path(f"norm_transfer_{seed}_decisions.npz")
    np.savez_compressed(bank_path, **bank)
    receipt = dict(seed=seed, decisions=sha(bank_path), f9=decision_sha, f9_proof=BEFORE_LABELS[seed],
                   actors=actors, active_entries=active, stock=owner.REPORT_SHA, data=f9.DATA_SHA,
                   orders=f9.PERMS_SHA, scorer=sha(Path("source.sha256")),
                   replay=sha(Path("recording-replay.log")))
    Path(f"norm_transfer_{seed}_before_labels.json").write_text(json.dumps(receipt, indent=2) + "\n")
    pairs = [("genuine", "stock"), ("genuine", "donor"), ("genuine", "f9_full")]
    check(sha(bank_path) == receipt["decisions"], "blind decision bank changed")
    with np.load(bank_path, allow_pickle=False) as stored:
        saved = {k: stored[k] for k in stored}
    metrics = f9.summarize(saved, pairs, meta, interval, metric)
    dest = Path(f"norm_transfer_{seed}_result.json")
    dest.write_text(json.dumps({**receipt, **metrics}, indent=2) + "\n")
    print(json.dumps({name: values["pp"] for name, values in metrics["absolute"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, choices=(460, 1460))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.smoke)
