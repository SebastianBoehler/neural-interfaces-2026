"""Authenticate the complete blind S1C replay before opening development labels."""
import argparse
import json
import os

import numpy as np

from replay_f13 import (ARCHIVE_SHA, DATA, DATA_SHA, HERE, ORDERS, ORDER_SHA, ORIGIN_SHA, PEOPLE,
                        ROOT, origin_report, row_sha, sha)


ARMS = ("original", "bounded")


def authenticate_held(array_id, report, subject, person):
    n = len(subject)
    actor = sha(HERE / "source.sha256")
    directory = HERE / "folds"
    expected_names = {f"{p}_s{s}.npz" for p in PEOPLE for s in (0, 1)}
    if {p.name for p in directory.iterdir()} != expected_names:
        raise ValueError("unsealed or incomplete blind decision folds")
    merged = {(name, seed): np.full((len(ORDERS), n), -1, dtype=np.int8)
              for name in ARMS for seed in (0, 1)}
    receipts, digests = {}, {}
    for seed in (0, 1):
        for i, held in enumerate(PEOPLE):
            key = f"{held}_s{seed}"
            prediction = directory / f"{key}.npz"
            digest = sha(prediction)
            log = (HERE / f"f13-{array_id}_{seed * len(PEOPLE) + i}.out").read_text().splitlines()
            if f"{digest}  {prediction.name}" not in log or "PASS F13 original parity and bounded normalization" not in log:
                raise ValueError(f"missing blind worker receipt for {key}")
            idx, fit = np.flatnonzero(person == held), np.flatnonzero(person != held)
            fold = ROOT / "output" / key / "result.npz"
            if sha(fold) != report["folds"][key]:
                raise ValueError(f"archived result mismatch for {key}")
            with np.load(fold, allow_pickle=False) as z:
                origin = json.loads(str(z["metadata"]))
                archived = np.stack([z[f"S1C|{order}"].argmax(1) for order in ORDERS])
                if not np.array_equal(z["index"], idx):
                    raise ValueError(f"archived recording rows differ for {key}")
            expected = dict(held=held, seed=seed, data=DATA_SHA, orders=ORDER_SHA, origin=ORIGIN_SHA,
                            archived_output=report["folds"][key], checkpoint=origin["checkpoints"]["S1"],
                            actor=actor, fit_people=sorted(set(subject[fit])), source_rows=row_sha(fit),
                            held_rows=row_sha(idx), held_labels_opened=False, original_labels_exact=True,
                            disabled_activations_exact=True)
            with np.load(prediction, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                if any(meta[k] != value for k, value in expected.items()):
                    raise ValueError(f"blind worker provenance differs for {key}")
                active = meta["active_past_channel_windows"]
                if (not np.array_equal(z["index"], idx) or not np.array_equal(z["original"], archived)
                        or "A100" not in meta["device"] or meta["first_windows_exact"] < len(ORDERS)
                        or active < 0 or bool(active) != meta["oracle_probe_exercised"]
                        or (not active and not np.array_equal(z["original"], z["bounded"]))):
                    raise ValueError(f"blind worker baseline or intervention differs for {key}")
                for arm in ARMS:
                    values = z[arm]
                    if values.shape != (len(ORDERS), len(idx)) or not np.isin(values, (0, 1, 2)).all():
                        raise ValueError(f"invalid {arm} decisions for {key}")
                    merged[arm, seed][:, idx] = values
            if json.dumps(meta, sort_keys=True) not in log:
                raise ValueError(f"blind worker log and fold receipts differ for {key}")
            receipts[key], digests[prediction.name] = meta, digest
    if any((v < 0).any() for v in merged.values()):
        raise ValueError("missing rows after blind merge")
    for seed in (0, 1):
        if not any(receipt["seed"] == seed and receipt["oracle_probe_exercised"]
                   for receipt in receipts.values()) or np.array_equal(merged["original", seed], merged["bounded", seed]):
            raise ValueError(f"no verified active bounded decision in seed {seed}")
    lock = HERE / "decision_banks.sha256"
    with lock.open("x") as handle:
        for name, digest in sorted(digests.items()):
            handle.write(f"{digest}  folds/{name}\n")
    return merged, receipts, sha(lock)


def score(array_id):
    report = origin_report()
    with np.load(DATA, allow_pickle=False) as z:
        subject = z["subject"].astype(str)
    person = np.array([value.split("/")[-1] for value in subject])
    merged, receipts, bank_sha = authenticate_held(array_id, report, subject, person)

    import protocol
    import score_s_v66p

    if (protocol.__file__ != str(ROOT / "experiments/protocol.py") or
            score_s_v66p.__file__ != str(ROOT / "studies/14nov/newmember/score_s_v66p.py")):
        raise ValueError("official metric imports differ from §323")
    data = protocol.load()
    y = data["y"]
    if not np.array_equal(data["subject"], subject):
        raise ValueError("metric subject rows differ")
    groups = [np.flatnonzero([order.startswith(s) for order in ORDERS]) for s in ("within", "full", "runs")]
    cells = np.char.add(np.char.add(data["stream"], "|"), data["interface"])
    counts = np.array([len(np.unique(cells[person == p])) for p in PEOPLE])
    seeds = {}
    for seed in (0, 1):
        tables = {}
        for arm in ARMS:
            decisions = merged[arm, seed]
            people = np.vstack([score_s_v66p.person_accuracy(np.eye(3)[pred], y, cells, person, PEOPLE)
                                for pred in decisions])
            official = np.array([100 * protocol.score(y, pred, data, np.ones(len(y), bool))[
                "balanced_accuracy_cell_mean"] for pred in decisions])
            if not np.allclose(official, np.average(people, weights=counts, axis=1), atol=1e-10, rtol=0):
                raise ValueError("paired tables differ from the official absolute score")
            tables[arm] = people
        delta = tables["bounded"] - tables["original"]
        seeded = {arm: score_s_v66p.cell_weighted_summary(tables[arm], ORDERS, counts) for arm in ARMS}
        seeded["paired"] = score_s_v66p.cell_weighted_summary(delta, ORDERS, counts)
        seeded["paired"]["orders"] = dict(zip(ORDERS, np.average(delta, weights=counts, axis=1).tolist()))
        seeded["paired"]["people"] = dict(zip(PEOPLE, np.mean([delta[g].mean(0) for g in groups], 0).tolist()))
        seeded["paired"]["changes"] = int(np.count_nonzero(merged["original", seed] != merged["bounded", seed]))
        seeds[str(seed)] = seeded
    historic = report["official_equal_cell_scheme_equal"]["S1C"]
    restored = float(np.mean([s["original"]["mean"] for s in seeds.values()]))
    if abs(restored - historic) > 1e-9:
        raise ValueError("reproduced S1C does not match the authentic historical result")
    limits = [seed["paired"]["interval"] for seed in seeds.values()]
    outcome = "undecided"
    if all(lo > 0.0 for lo, _ in limits):
        outcome = "supports positive contrast"
    elif all(hi <= 0.0 for _, hi in limits):
        outcome = "refutes positive contrast"
    payload = dict(experiment="F13", scope="held-new WHOLE DEV", source_array=array_id,
                   source=sha(HERE / "source.sha256"), scorer=sha(HERE / "score.sha256"), origin=ORIGIN_SHA,
                   historical_result=ARCHIVE_SHA, historical_s1c=historic, reproduced_s1c=restored,
                   decision_banks=bank_sha, orders=list(ORDERS), n_cells=counts.tolist(), seeds=seeds,
                   fold_receipts=receipts, outcome=outcome)
    result = HERE / "whole_results.json"
    tmp = result.with_suffix(".tmp")
    with tmp.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.link(tmp, result)
    tmp.unlink()
    print(f"{sha(result)}  {result.name}", flush=True)
    print("PASS F13 blind decisions and official scoring", flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("array_id", type=int)
    score(args.parse_args().array_id)
