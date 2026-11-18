"""Label-aware complementarity ceiling of exact external decision banks."""

import hashlib
import json
from pathlib import Path

import numpy as np

from data import permutation
from protocol import score


ROOT = Path("/mnt/lustre/work/gehler/gfh527")
ARCHIVE = ROOT / "ni26_qualify385_v1_20261118"
DATA = ROOT / "neural-interfaces-2026/data/scherer2015/windows.npz"
C3 = ROOT / "ni26_friday_c3_v1_20261118"
C4 = ROOT / "ni26_friday_c4_v1_20261118"
KERNEL = ROOT / "ni26_external459_v2_20261118"
INPUTS = {
    ARCHIVE / "inputs.sha256": "581227e7afd46b179d46231ae99366c1763ac901fc4b7569b69230d07166748f",
    DATA: "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91",
    C3 / "before_labels.json": "5453a86cf856ca844534b5638bb45c69b671af40cc98bad11c746f2ca9a5b7b2",
    C4 / "before_labels.json": "245e2510c7c05cf4fb8ec4ec5ff61c277cb78859b9660287d8f1a062fab71615",
    KERNEL / "external_before_labels.json": "404cebaf87671a8c36ff2565b6719381bdac24bd215f5fdd42b6f7d134b6b280",
}
ARMS = ("stock", "consistency", "balanced", "kernel454", "kernel1454")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, expected):
    if sha(path) != expected:
        raise ValueError(f"decision hash mismatch {path}")
    return json.loads(path.read_text())


def collect(subjects, days, streams, people, visits):
    orders = [permutation(streams, None, "full", seed) for seed in (0, 1)]
    pred = np.full((len(ARMS), len(orders), len(subjects)), -1, np.int8)
    c3, c4, kernel = [read(path, INPUTS[path]) for path in (
        C3 / "before_labels.json", C4 / "before_labels.json", KERNEL / "external_before_labels.json"
    )]
    for i, person in enumerate(people):
        name = f"primary_{i}.json"
        consistency = read(C3 / name, c3[name])
        balanced = read(C4 / name, c4[name])
        archive = ARCHIVE / f"cap_{i}.json"
        name = f"external_0_{i}.json"
        retrieval = read(KERNEL / name, kernel["actors"][name])
        official = read(archive, retrieval["archive"])
        assert retrieval["source"] == kernel["source"] and retrieval["models"] == kernel["models"]
        assert set(official) == set(consistency) == set(balanced) == set(retrieval["sessions"])
        for oi, perm in enumerate(orders):
            for day in visits[person]:
                rows = perm[(subjects == person) & (days == day)]
                key = f"{person}|{day}|{oi}"
                stock, c3_row, c4_row = (item[key] for item in (official, consistency, balanced))
                other = retrieval["sessions"][key]
                assert rows.tolist() == stock["rows"] == c3_row["rows"] == c4_row["rows"] == other["rows"]
                assert stock["activity"] == c3_row["activity"] == c4_row["activity"] == other["counters"]
                assert stock["labels"] == other["arms"]["stock"]
                assert (pred[:, oi, rows] == -1).all()
                pred[:, oi, rows] = [stock["labels"], c3_row["labels"], c4_row["labels"],
                                      other["arms"]["454_memory"], other["arms"]["1454_memory"]]
    assert np.isin(pred, [0, 1, 2]).all()
    bank = Path("d1_decisions.npz")
    np.savez_compressed(bank, arms=ARMS, predictions=pred)
    Path("d1_before_labels.json").write_text(json.dumps(dict(
        inputs={str(k): v for k, v in INPUTS.items()}, decisions=sha(bank),
        source=sha(Path("d1.sha256"))), indent=2) + "\n")
    return pred


def main():
    assert sha(DATA) == INPUTS[DATA] and sha(ARCHIVE / "inputs.sha256") == INPUTS[ARCHIVE / "inputs.sha256"]
    assert Path("replay.log").read_text().splitlines()[-1] == "OK"
    with np.load(DATA, allow_pickle=False) as source:
        subjects, days, interfaces = (source[key] for key in ("subject", "session", "interface"))
    streams = np.char.add(np.char.add(subjects, "|"), days)
    people = sorted(set(subjects))
    visits = {person: sorted(set(days[subjects == person])) for person in people}
    assert len(subjects) == 2130 and len(people) == 9 and all(len(v) == 2 for v in visits.values())
    pred = collect(subjects, days, streams, people, visits)
    with np.load(DATA, allow_pickle=False) as source:
        labels = source["y"]
    stock = pred[0] == labels
    fixes = (pred[1:] == labels) & ~stock
    union = fixes.any(axis=0)
    oracle = np.where(union, labels, pred[0])
    meta = dict(stream=streams, interface=interfaces)
    full = np.ones(len(labels), bool)
    stock_scores, oracle_scores = [], []
    for oi in range(2):
        for values, acc in ((pred[0], stock_scores), (oracle, oracle_scores)):
            cell = score(labels, values[oi], meta, full)
            assert cell["n_cells"] == 18
            acc.append(100 * cell["balanced_accuracy_cell_mean"])
    stock_pp = float(np.mean(stock_scores))
    assert abs(stock_pp - 48.45568783068782) < 1e-10
    delta = np.empty((9, 2, 2))
    for i, person in enumerate(people):
        for j, day in enumerate(visits[person]):
            masks = [(subjects == person) & (days == day) & (labels == c) for c in range(3)]
            assert all(mask.any() for mask in masks)
            delta[i, j] = 100 * np.mean([union[:, mask].mean(axis=1) for mask in masks], axis=0)
    assert abs(delta.mean() - (np.mean(oracle_scores) - stock_pp)) < 1e-10
    rng = np.random.default_rng(459)
    pp = rng.integers(9, size=(10000,9,1,1))
    dd = rng.integers(2, size=(10000,9,2,1))
    oo = rng.integers(2, size=(10000,9,2,2))
    result = dict(stock=stock_pp, oracle=float(np.mean(oracle_scores)), pp=float(delta.mean()),
                  paired80=np.quantile(delta[pp,dd,oo].mean((1,2,3)), [.1,.9]).tolist(),
                  rescued=int(union.sum()), stock_errors=int((~stock).sum()),
                  individually_rescued=dict(zip(ARMS[1:], fixes.sum((1,2)).tolist())),
                  uniquely_rescued=dict(zip(ARMS[1:], (fixes & (fixes.sum(0) == 1)).sum((1,2)).tolist())),
                  people=dict(zip(people, delta.mean((1,2)).tolist())),
                  orders=delta.mean((0,1)).tolist(), proof=sha(Path("d1_before_labels.json")))
    Path("d1_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
