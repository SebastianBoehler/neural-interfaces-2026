"""Pair guarded three-seed outputs with §419 predictions on identical people and orders."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "runs/aux430"
spec = importlib.util.spec_from_file_location("lineage", ROOT / "studies/18nov/lineage419/score.py")
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)


def source_folds():
    """Reconstruct the producer row masks on the original metadata."""
    sys.path.insert(0, str(ROOT / "experiments"))
    import protocol
    with np.load(protocol.DATA) as data:
        rows = {k: data[k] for k in ("subject", "split", "task")}
    graz = np.flatnonzero(np.char.endswith(rows["task"].astype(str), "Graz"))
    folded = {}
    for name, train, test in protocol.folds(rows, "all_loso"):
        subjects = rows["subject"]
        assert np.intersect1d(subjects[train], subjects[test]).size == 0
        assert np.intersect1d(subjects[graz[train[graz]]], subjects[test]).size == 0
        assert np.unique(subjects[train]).size == 9
        assert np.unique(subjects[test]).tolist() == [f"Dreyer2026Proteus/{name}"]
        folded[name] = dict(train=int(train.sum()), held=int(test.sum()), train_rest=int(train[graz].sum()))
    assert len(folded) == 10 and sum(f["held"] for f in folded.values()) == 2655
    return folded


def main():
    assert (WORK / "replay.txt").read_text().splitlines()[-1] == "OK"
    three, hashes, flipped = [], {}, {}

    def add(protocol, order, rows, perm, labels, cells, owners, old, historical):
        assert protocol == "all_loso"
        path = WORK / f"{order}.npz"
        hashes[order] = lineage.metric.sha256(path)
        with np.load(path) as data:
            candidate = data["scores"]
            assert candidate.shape == old.shape == historical.shape == (2655, 3)
            assert str(data["raw_sha"]) == lineage.metric.sha256(lineage.STORE / f"s281_v66p_control_{order}.npz")
            assert str(data["cp_sha"]) == lineage.metric.sha256(lineage.STORE / f"control_{order}_B2_pool.npz")
            assert float(data["max_eye_log_change"]) > 0
            assert (float(data["max_rest_change"]) > 0) == (order in ("within0", "within1"))
        flipped[order] = int(np.sum(candidate.argmax(1) != historical.argmax(1)))
        three.append(lineage.metric.person_accuracy(candidate, labels, cells, owners, lineage.PEOPLE[protocol]))

    proto, old, historical, weights, n = next(lineage.guarded_matrices(observe_order=add))
    assert proto == "all_loso" and n == 2655 and weights.sum() == 28 and len(three) == 8
    inputs = [(old, historical, "historical_minus_old"), (old, three, "three_minus_old"), (historical, three, "three_minus_historical")]
    report = {name: {k: lineage.summary(a, b, w) for k, w in
                     (("official", weights), ("equal_person", np.ones(len(weights))))} for a, b, name in inputs}
    owner = json.loads((ROOT / "evidence/next_wave_16nov/lineage419_result.json").read_text())
    assert report["historical_minus_old"]["official"] == owner["results"][proto]["official"]
    report.update(orders=lineage.metric.ORDERS, people=lineage.PEOPLE[proto], rows=n,
                  cell_weights=weights.tolist(), changed_labels_by_order=flipped, three_sha256=hashes, reconstructed_fit_rows=source_folds())
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
