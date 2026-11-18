"""Audit the paired §419 and §411b endpoints before reporting signed flip contributions."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from metrics import decompose, describe, position_bins, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = ROOT / "evidence/next_wave_16nov"
BANK = ROOT / "runs/timeline427/bank"
spec = importlib.util.spec_from_file_location("lineage", HERE.parent / "lineage419/score.py")
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)
sys.path.insert(0, str(ROOT / "studies/17nov/external_pair352"))
from data import permutation


def load(path):
    return json.loads(path.read_text())


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)


def public():
    locked = load(EVIDENCE / "lineage419_result.json")
    matrices = {p: [] for p in lineage.PEOPLE}
    with np.load(lineage.RUN / "replay_data/bases.npz", allow_pickle=True) as z:
        streams = {p: z["stream|" + p] for p in matrices}

    def observe(protocol, order_name, rows, order, y, cells, owners, old, cap):
        if order_name != locked["orders"][len(matrices[protocol])]:
            raise ValueError("unexpected public order")
        bins = position_bins(rows, order, streams[protocol])
        matrix, _ = decompose(y, cells, owners, lineage.PEOPLE[protocol], bins,
                              old.argmax(1), cap.argmax(1))
        matrices[protocol].append(matrix)

    groups = {s: [i for i, o in enumerate(locked["orders"]) if o.startswith(s)]
              for s in ("within", "full", "runs")}
    result, points = {}, {}
    for name, old, cap, w, _ in lineage.guarded_matrices(observe):
        ref = locked["results"][name]["official"]
        for scores, endpoint in [(old, "old"), (cap, "new")]:
            by_person = np.mean([scores[g].mean(0) for g in groups.values()], axis=0)
            close(by_person, ref[endpoint + "_by_person"])
        m = np.array(matrices[name])
        close((m[:, :, 1] - m[:, :, 2]).sum((-2, -1)), cap-old)
        result[name], points[name] = summarize(m, w, lineage.PEOPLE[name], groups)
        close(result[name]["delta_pp"], ref["delta"])
        if name == "seen_session":
            result["returning_20_80"], points["returning_20_80"] = summarize(
                m, np.array([.2, .8]), lineage.PEOPLE[name], groups)
    if lineage.LOCK != locked["sha256"]:
        raise ValueError("§419 inputs differ from the owner manifest")
    result["mixture"] = describe(.566 * points["all_loso"] + .434 * points["returning_20_80"])
    return result


def external(smoke=False):
    path = ROOT / "data/scherer2015/windows.npz"
    checks = {path: "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91",
              BANK/"result.json": "df4a38dc2e13bdb58d8ad44e914dbb6d2e22d0ee44e74e9e4be4388263073ecd",
              BANK/"output.sha256": "b68b92145a5c01691b244aab1600714231ac6f7fb85e894774a6384f91d9b857",
              ROOT/"studies/17nov/external_pair352/data.py": "10ab487a8c3bbf588a4ef10ced6834f28a58da8dddac355e83185b18214185b5"}
    for line in (BANK/"output.sha256").read_text().splitlines():
        digest, filename = line.split()
        checks[BANK/filename] = digest
    for source, digest in checks.items():
        if lineage.metric.sha256(source) != digest:
            raise ValueError(f"unverified source {source}")
    with np.load(path) as z:
        owners, sessions, y, interfaces = (z[k] for k in ("subject", "session", "y", "interface"))
    streams = np.char.add(np.char.add(owners, "|"), sessions)
    cells = np.char.add(np.char.add(streams, "|"), interfaces)
    ref = load(EVIDENCE/"cross411b_package.json")
    activity = load(BANK/"result.json")["activity"]
    people = ref["people"]
    rows = np.arange(len(y))
    matrices, abs_scores, class_scores = [], [], []
    for seed in (411, 1411):
        order = permutation(streams, None, "full", seed)
        bins = position_bins(rows, order, streams)
        preds = []
        for arm in ("old", "cap"):
            pred = np.full(len(y), -1)
            for i, person in enumerate(people):
                records = load(BANK/f"{arm}_{i}.json")
                for stream in np.unique(streams[owners == person]):
                    rec = records[f"{stream}|{seed}"]
                    expected = order[streams[order] == stream]
                    np.testing.assert_array_equal(expected, rec["rows"])
                    if activity[f"{arm}|{stream}|{seed}"] != rec["activity"]:
                        raise ValueError("worker activity differs")
                    pred[expected] = rec["labels"]
            preds.append(pred)
        matrix, w = decompose(y, cells, owners, people, bins, *preds)
        a = np.array([lineage.metric.person_accuracy(np.eye(3)[p], y, cells, owners, people) for p in preds])
        close((matrix[:, 1]-matrix[:, 2]).sum((-2,-1)), a[1]-a[0])
        if smoke:
            return {"smoke_seed": seed, "verified_people": len(people)}
        matrices.append(matrix)
        abs_scores.append(a)
        class_scores.append([[np.mean([100*(p[(cells == c) & (y == k)] == k).mean()
                                       for c in np.unique(cells)]) for k in range(3)] for p in preds])
    abs_scores = np.array(abs_scores)
    close(np.average(abs_scores.mean(0), axis=1, weights=w), ref["acc"])
    close(np.mean(class_scores, axis=0), ref["classes"])
    delta = abs_scores[:, 1]-abs_scores[:, 0]
    close(delta.mean(0), ref["person_diff"])
    close(np.average(delta, axis=1, weights=w), ref["order_diff"])
    return summarize(np.array(matrices), w, people, {"full": [0, 1]})[0]


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--smoke", action="store_true")
    args = cli.parse_args()
    if (ROOT/"runs/timeline427/replay.txt").read_text().splitlines()[-1] != "OK":
        raise ValueError("replay/check.py must pass")
    report = external(smoke=args.smoke)
    if not args.smoke:
        report = dict(public=public(), exact_external=report)
    print(json.dumps(report, indent=2))
