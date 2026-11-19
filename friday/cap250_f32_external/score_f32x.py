"""Authenticate the complete exact native banks, then score the official cells."""
import json
from pathlib import Path

import numpy as np

from fixture import ARMS, DATA_SHA, SEEDS, schedule
from protocol import score
from score import DATA, anchor_path, changes, seal
from settings import sha, stamp

ROWS = 2130


def qualify():
    assert sha(DATA) == DATA_SHA
    source, actors = sha("source.sha256"), sha("actors.json")
    smoke_file = Path("smoke.json")
    smoke = json.loads(smoke_file.read_text())
    assert smoke["passed"] and smoke["mode"] == "smoke" and not smoke["held_labels_opened"]
    assert smoke["source"] == source and smoke["actors"] == actors
    assert smoke["results"]["P10-3"]["stop"] == 10
    assert all(r["native_zero_exact"] and r["exact_private_prefix"] for r in smoke["results"].values())
    with np.load(DATA, allow_pickle=False) as z:
        subjects, visits, interfaces = (z[k] for k in ("subject", "session", "interface"))
    streams = np.char.add(np.char.add(subjects, "|"), visits)
    roster = sorted(set(subjects))
    assert all(len(set(interfaces[streams == key])) == 1 for key in set(streams))
    plans = schedule(subjects, visits)
    bank = np.full((len(ARMS), len(SEEDS), ROWS), -1, dtype=np.int8)
    signed, summaries = {}, {}
    for index, who in enumerate(roster):
        worker = Path(f"person_{index}.json")
        pupil = Path(f"pupils_produce_{index}.npz")
        assert not (worker.stat().st_mode | pupil.stat().st_mode) & 0o222
        r = json.loads(worker.read_text())
        old = anchor_path(index)
        anchor = json.loads(old.read_text())["sessions"][f"{who}|1|0"]
        assert r["passed"] and r["mode"] == "produce" and not r["held_labels_opened"]
        assert r["source"] == source and r["actors"] == actors and r["archive"] == sha(old)
        assert r["pupil"] == sha(pupil) and r["index"] == index and r["who"] == who
        assert r["cold_rows"] == anchor["rows"] and r["cold"]["labels"] == anchor["labels"]
        assert r["cold"]["activity"] == anchor["activity"] and "RTX 2080 Ti" in r["device"]
        assert len(r["results"]) == 4
        signed[worker.name] = sha(worker)
        summaries[who] = {"warm": 0}
        for n, seed in enumerate(SEEDS):
            for date in (1, 2):
                stream = f"{who}|{date}"
                rows = plans[seed][stream]
                stats = r["results"][f"{stream}|{seed}"]
                act = stats["activity"]
                assert stats["rows"] == rows.tolist() and stats["native_zero_exact"] and stats["stop"] is None
                assert (bank[:, n, rows] == -1).all() and act["windows"] == len(rows) and act["recognized"] == -1
                assert act["ft_steps"] == 20*act["ft_refits"] and act["errors"] == act["ft_over"] == 0
                assert stats["owner"]["refit_ages"] == list(range(20,len(rows),10))
                assert min(stats["maximum_mix"].values()) > 1e-6 and stats["warm"] > 0
                assert set(stats["stages"]) == set(ARMS[1:])
                for stage in stats["stages"].values():
                    assert stage["fit_ages"] == list(range(20,len(rows),20)) and stage["fusions"] > 0
                for a, arm in enumerate(ARMS):
                    guess = stats[arm]
                    assert len(guess) == len(rows) and np.isin(guess, (0, 1, 2)).all()
                    bank[a, n, rows] = guess
                summaries[who]["warm"] += stats["warm"]
    assert (bank >= 0).all()
    lock = seal(Path("before_labels.json"), dict(source=source, scoring=sha("scoring.sha256"),
                                                  workers=signed, actors=actors, smoke=sha(smoke_file),
                                                  predictions=stamp(bank), people=roster, orders=SEEDS,
                                                  held_labels_opened=False))
    return bank, subjects, visits, interfaces, streams, roster, lock, summaries


def official_cells(bank, gold, subject, visit, meta, people):
    classes = np.zeros((len(ARMS), len(people), 2, len(SEEDS), 3))
    for i, who in enumerate(people):
        for v, date in enumerate(("1", "2")):
            for label in (0, 1, 2):
                select = (subject == who) & (visit == date) & (gold == label)
                assert select.any()
                classes[:, i, v, :, label] = 100*np.mean(bank[:, :, select] == label, axis=-1)
    cells = classes.mean(-1)
    for i in range(len(ARMS)):
        for j in range(len(SEEDS)):
            official = score(gold, bank[i, j], meta, np.ones(ROWS, bool))
            assert official["n_cells"] == 18
            np.testing.assert_allclose(np.sort(100*np.asarray(official["cell_scores"])),
                                       np.sort(cells[i, :, :, j].ravel()), atol=1e-10, rtol=0)
    return cells, classes


def main():
    bank, subject, visit, interface, streams, people, lock, summaries = qualify()
    with np.load(DATA, allow_pickle=False) as z:
        y = z["y"]
    cells, classes = official_cells(bank, y, subject, visit, dict(stream=streams, interface=interface), people)
    rng = np.random.default_rng(468)
    i = rng.integers(9, size=(10000, 9, 1, 1))
    v = rng.integers(2, size=(10000, 9, 2, 1))
    o = rng.integers(2, size=(10000, 9, 2, 2))

    def contrast(delta):
        bounds = np.percentile(delta[i,v,o].mean(axis=(1,2,3)), [10,90]).tolist()
        orders = dict(zip(SEEDS, delta.mean(axis=(0,1)).tolist(), strict=True))
        verdict = "supported" if bounds[0] > 0 and min(orders.values()) > 0 else "refuted" if bounds[1] <= 0 else "undecided"
        return dict(pp=float(delta.mean()), paired80=bounds, order_pp=orders, verdict=verdict)

    outcomes = dict(joint_minus_stock=contrast(cells[2]-cells[0]), joint_minus_hard=contrast(cells[2]-cells[1]),
                    joint_minus_hard_motor=contrast(classes[2,...,0]-classes[1,...,0]),
                    hard_minus_stock=contrast(cells[1]-cells[0]))
    payload = dict(scope="FULL-SYSTEM EXTERNAL conditional on identical live native RTX upstream",
                   before_labels=lock, absolute_pct=dict(zip(ARMS, cells.mean(axis=(1,2,3)).tolist(), strict=True)),
                   contrasts=outcomes, joint_vs_stock=changes(bank[[0,2]], y, np.ones(ROWS, bool)),
                   joint_vs_hard=changes(bank[[1,2]], y, np.ones(ROWS, bool)),
                   person_pp=dict(zip(people, (cells[2]-cells[1]).mean((1,2)).tolist(), strict=True)),
                   class_pp=(classes[2]-classes[1]).mean((0,1,2)).tolist(), active=summaries)
    digest = seal(Path("result.json"), payload)
    print("PASS F32x official full-system score", digest, outcomes, flush=True)


if __name__ == "__main__":
    main()
