"""Quantify only registered decision discrepancies after §451 is sealed."""
from pathlib import Path
import hashlib
import json

import numpy as np

BASE = Path("/mnt/lustre/work/gehler/gfh527")
ARCHIVE = BASE / "ni26_qualify385_v1_20261118"
NATIVE = BASE / "ni26_native451_v1_20261118"
DATA = BASE / "neural-interfaces-2026/data/scherer2015/windows.npz"
DATA_SHA = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"


def read(path):
    return json.loads(path.read_text())


def exact():
    return dict(native_minus_python_pp=0., corrections=0, regressions=0,
                basis="identical decisions, independently of task labels")


def metric(records, gold, people, days, subject, day):
    archive, python, native = records
    rows = np.asarray(archive["rows"])
    assert all(r["rows"] == archive["rows"] for r in records)
    expected = np.flatnonzero((people == subject) & (days == day))
    assert np.array_equal(expected, np.sort(rows))
    labels = gold[rows]
    assert set(labels.tolist()) == {0, 1, 2}
    preds = np.array([r["labels"] for r in records])
    assert preds.shape == (3, len(rows)) and np.isin(preds, [0, 1, 2]).all()
    hits = preds == labels
    scores = np.mean([hits[:, labels == cls].mean(axis=1) for cls in range(3)], axis=0) * 100
    return scores, int((hits[2] & ~hits[1]).sum()), int((hits[1] & ~hits[2]).sum())


def main():
    result = read(NATIVE / "result.json")
    assert "OK" in (NATIVE / "replay-proof.txt").read_text().splitlines()
    assert len(result["input_hashes"]) == 18
    for name, digest in result["input_hashes"].items():
        assert hashlib.sha256((NATIVE / name).read_bytes()).hexdigest() == digest
    totals = result["totals"]
    assert totals["windows"] == 4260 and len(result["streams"]) == 36
    repeat = result["warm_first_stream"]
    assert len(repeat) == 2
    output = dict(primary=exact(), warm={key: exact() for key in repeat},
                  bank_sha256=hashlib.sha256((NATIVE / "result.json").read_bytes()).hexdigest())
    score_primary = bool(totals["disagreements"] or totals["python_drift"])
    score_warm = {k for k, v in repeat.items() if v["paired"]["labels"]}
    if score_primary or score_warm:
        assert hashlib.sha256(DATA.read_bytes()).hexdigest() == DATA_SHA
        with np.load(DATA) as z:
            gold, people, days = (z[k] for k in ("y", "subject", "session"))
        cells = np.full((9, 2, 2, 3), np.nan)
        corrections = regressions = 0
        for i, person in enumerate(sorted(set(people))):
            archive = read(ARCHIVE / f"cap_{i}.json")
            python, native = [read(NATIVE / f"full_{arm}_{i}.json") for arm in ("python", "native")]
            for key, record in archive.items():
                subject, day, seed = key.split("|")
                assert subject == person
                if score_primary:
                    scores, c, r = metric((record, python[key], native[key]), gold, people, days, subject, day)
                    cells[i, sorted(set(days[people == person])).index(day), int(seed)] = scores
                    corrections += c
                    regressions += r
                if key in score_warm:
                    scores, c, r = metric((record, python["repeat"], native["repeat"]),
                                          gold, people, days, subject, day)
                    output["warm"][key] = dict(native_minus_python_pp=float(scores[2] - scores[1]),
                                               archive_python_native=scores.tolist(), corrections=c, regressions=r)
        if score_primary:
            assert np.isfinite(cells).all()
            scores = cells.mean((0, 1, 2))
            assert np.isclose(scores[0], read(ARCHIVE / "result.json")["acc"][1])
            diff = cells[..., 2] - cells[..., 1]
            rng = np.random.default_rng(385)
            i = rng.integers(9, size=(10000, 9, 1, 1))
            j = rng.integers(2, size=(10000, 9, 2, 1))
            k = rng.integers(2, size=(10000, 9, 2, 2))
            output["primary"] = dict(native_minus_python_pp=float(diff.mean()), archive_python_native=scores.tolist(),
                                     ci80=np.quantile(diff[i, j, k].mean((1, 2, 3)), [.1, .9]).tolist(),
                                     corrections=corrections, regressions=regressions)
    Path("result.json").write_text(json.dumps(output, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
