"""Authenticate the RTX/A100 pair and score changes on the same Scherer rows."""

import hashlib
import json
from pathlib import Path

import numpy as np

from data import permutation
from protocol import score


ROOT = Path("/mnt/lustre/work/gehler/gfh527")
PRIOR = ROOT / "ni26_qualify385_v1_20261118"
DATA = ROOT / "neural-interfaces-2026/data/scherer2015/windows.npz"
ARMS = ("old", "cap")
PEOPLE = (0, 5)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert Path("replay.log").read_text().splitlines()[-1] == "OK"
    assert sha(DATA) == "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"
    with np.load(DATA, allow_pickle=False) as source:
        people, days, interfaces = (source[k] for k in ("subject", "session", "interface"))
    subjects = sorted(set(people))
    cohort = [subjects[i] for i in PEOPLE]
    assert cohort == ["Scherer2015/A", "Scherer2015/G"]
    mask = np.isin(people, cohort)
    sessions = np.char.add(np.char.add(people, "|"), days)
    orders = [permutation(sessions, None, "full", seed) for seed in (0, 1)]
    pred = np.full((2, 2, 2, len(people)), -1, np.int8)
    record, receipts, changes = {}, {}, {}
    for ai, arm in enumerate(ARMS):
        for person in PEOPLE:
            path = Path(f"rtx_{arm}_{person}.json")
            pinned = PRIOR / f"{arm}_{person}.json"
            receipts[path.name] = sha(path)
            actual = json.loads(path.read_text())
            expected = json.loads(pinned.read_text())
            who = subjects[person]
            assert actual["arm"] == arm and actual["person"] == person and "RTX 2080 Ti" in actual["device"]
            assert set(actual["sessions"]) == set(expected) and len(expected) == 4
            timing, differing = [], []
            for oi, order in enumerate(orders):
                for day in sorted(set(days[people == who])):
                    rows = order[(people == who) & (days == day)]
                    key = f"{who}|{day}|{oi}"
                    prior, observed = expected[key], actual["sessions"][key]
                    assert rows.tolist() == prior["rows"] == observed["rows"]
                    assert prior["activity"] == observed["activity"]
                    assert (pred[:, ai, oi, rows] == -1).all()
                    pred[0, ai, oi, rows] = prior["labels"]
                    pred[1, ai, oi, rows] = observed["labels"]
                    labels = zip(rows.tolist(), prior["labels"], observed["labels"])
                    differing.extend(dict(session=key, row=r, a100=x, rtx=y) for r, x, y in labels if x != y)
                    timing.append(dict(session=key, seconds=observed["seconds"],
                                       copy_seconds=observed["copy_seconds"]))
            record.setdefault(who, {})[arm] = dict(times=timing, load_seconds=actual["load_seconds"],
                peak_cuda_reserved_bytes=actual["peak_cuda_reserved_bytes"], device=actual["device"])
            changes.setdefault(who, {})[arm] = differing
    bank = pred[:, :, :, mask]
    assert bank.shape == (2, 2, 2, 480) and np.isin(bank, [0, 1, 2]).all()
    path = Path("rtx_decisions.npz")
    np.savez_compressed(path, decisions=bank)
    proof = Path("rtx_before_labels.json")
    proof.write_text(json.dumps(dict(inputs=receipts, predictions=sha(path),
                    source=sha(Path("rtx_source.sha256")), parents=sha(Path("rtx_parents.sha256"))), indent=2))
    with np.load(DATA, allow_pickle=False) as source:
        labels = source["y"]
    meta = dict(stream=sessions, interface=interfaces)
    accuracies = {}
    for hi, hardware in enumerate(("A100", "RTX2080Ti")):
        scores = {}
        for ai, arm in enumerate(ARMS):
            official = [score(labels[mask], values, meta, mask) for values in bank[hi, ai]]
            assert all(run["n_cells"] == 4 for run in official)
            scores[arm] = [100 * run["balanced_accuracy_cell_mean"] for run in official]
        accuracies[hardware] = dict(scores=scores, delta_pp=float(np.mean(scores["cap"]) - np.mean(scores["old"])))
    warm_ratio = {}
    for who, arms in record.items():
        warm = {arm: sum(v["seconds"] for v in value["times"][1:]) for arm, value in arms.items()}
        warm_ratio[who] = warm["cap"] / warm["old"]
    drift = accuracies["RTX2080Ti"]["delta_pp"] - accuracies["A100"]["delta_pp"]
    result = dict(accuracy=accuracies, drift_pp=drift, changed=changes, times=record,
                  agreement_pass=all(len(v) <= 2 for pair in changes.values() for v in pair.values()),
                  effect_pass=abs(drift) <= .30, warm_ratio=warm_ratio, proof=sha(proof))
    Path("rtx_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(accuracy=accuracies, drift_pp=drift,
                          counts={p: {k: len(z) for k,z in v.items()} for p,v in changes.items()},
                          warm_ratio=warm_ratio)))


if __name__ == "__main__":
    main()
