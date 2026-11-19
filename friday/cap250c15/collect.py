"""Verify and score the three full-system rebalancing trajectories."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from bank_tools import archive_parity, seal, sha
from metrics import boot_draws, counts, interval, timing
from protocol import score
from settings import ARMS, SEEDS, STOCK_WORK, schedule


BASE = Path("/mnt/lustre/work/gehler/gfh527")
DATA = BASE / "neural-interfaces-2026/data/scherer2015/windows.npz"
PRIOR = BASE / "ni26_cross411b_20261118"
ROWS = 2130


def read(path):
    return json.loads(path.read_text())


def authenticated(path):
    digest = sha(path)
    assert path.with_suffix(".sha256").read_text().strip() == digest
    return read(path), digest


def verify(source):
    assert source == sha("source.sha256")
    smoke, smoke_digest = authenticated(Path("smoke.json"))
    assert smoke["passed"] and smoke["source"] == source and not smoke["labels_opened"]
    assert smoke["recognized_public"]["result"]["stock"]["activity"]["recognized"] == 10
    with np.load(DATA, allow_pickle=False) as f:
        people, days, iface = (f[k] for k in ("subject", "session", "interface"))
    stream = np.char.add(np.char.add(people, "|"), days)
    assert all(len(set(iface[stream == s])) == 1 for s in set(stream))
    roster = sorted(set(people))
    assert roster == read(PRIOR / "result.json")["people"]
    deliveries = schedule(people, days)
    matrix = np.full((len(ARMS), len(SEEDS), ROWS), -1, dtype=np.int8)
    runtime = np.zeros((len(ARMS), len(roster), 2, len(SEEDS)))
    manifest, old = {}, {}
    work = {a: Counter() for a in ARMS[1:]}
    effects = {a: Counter() for a in ARMS[1:]}
    for i, person in enumerate(roster):
        visits = [f"{person}|{v}" for v in (1, 2)]
        old_path = PRIOR / f"cap_{i}.json"
        archive = read(old_path)
        old[old_path.name] = sha(old_path)
        file = Path(f"person_{i}.json")
        bank, manifest[file.name] = authenticated(file)
        assert (bank["who"], bank["index"], bank["source"], bank["smoke_sha"], bank["archive"],
                bank["labels_opened"]) == (person, i, source, smoke_digest, old[old_path.name], False)
        anchors = bank["archived_controls"]
        assert set(anchors) == {f"{visits[0]}|{seed}" for seed in (411, 1411)}
        for key, value in anchors.items():
            assert value["rows"] == archive[key]["rows"]
            archive_parity(value["run"], archive[key])
        assert set(bank["sessions"]) == {f"{v}|{s}" for v in visits for s in SEEDS}
        for j, seed in enumerate(SEEDS):
            for d, visit in enumerate(visits):
                rows = deliveries[seed][visit]
                record = bank["sessions"][f"{visit}|{seed}"]
                rotation = (i + 2*j + d) % 3
                assert record["order"] == list(ARMS[rotation:]+ARMS[:rotation])
                assert np.array_equal(record["rows"], rows) and (matrix[:, j, rows] == -1).all()
                assert set(record["arms"]) == set(ARMS)
                stock = record["arms"]["stock"]
                for a, arm in enumerate(ARMS):
                    value = record["arms"][arm]
                    act, stat = value["activity"], value["receipt"]
                    assert len(value["labels"]) == len(rows) and np.isin(value["labels"], (0, 1, 2)).all()
                    assert np.isfinite(value["seconds"]) and value["seconds"] > 0
                    assert act["windows"] == len(rows) and act["errors"] == act["ft_over"] == 0
                    assert act["ft_steps"] == 20*act["ft_refits"]
                    assert [act[k] for k in STOCK_WORK] == stock["protected_work"] == value["protected_work"]
                    drift = value["protected_max"].copy()
                    assert drift.pop("native_warm_individual") <= 1e-3
                    assert max(drift.values()) <= 2e-5 and value["refit_eeg_sha"] == stock["refit_eeg_sha"]
                    assert stat["calls"] == stat["em_calls"] == len(rows)
                    assert stat["rb_calls"] == (0 if arm == "stock" else len(rows))
                    assert stat["exposure_max_error"] <= 2e-12
                    matrix[a, j, rows] = value["labels"]
                    runtime[a, i, d, j] = value["seconds"]
                    if arm == "stock":
                        continue
                    work[arm].update(k for k in act if act[k] != stock["activity"][k])
                    for k in ("calls", "above_001", "clipped_history_occurrences"):
                        effects[arm][k] += stat[k]
                    for k in ("exposure_max_error", "max_direction_delta", "max_stock_posterior_delta",
                              "max_scalar_posterior_delta"):
                        effects[arm][k] = max(effects[arm][k], stat[k])
    assert (matrix >= 0).all() and (runtime > 0).all()
    assert all(effects[a]["calls"] == ROWS*len(SEEDS) and effects[a]["above_001"] > 0 for a in work)
    proof = dict(source=source, smoke_sha=smoke_digest, roster=roster, seeds=SEEDS, arms=ARMS,
                 worker_receipts=manifest, archived_controls=old, activity_differences=work, effects=effects,
                 decisions_sha=hashlib.sha256(matrix.tobytes()).hexdigest(), labels_opened=False)
    return matrix, runtime, proof, people, days, dict(stream=stream, interface=iface)


def cells_for(y, matrix, people, days, meta, roster):
    by_class = np.empty((3, 9, 2, 2, 3))
    for i, who in enumerate(roster):
        for v, visit in enumerate(("1", "2")):
            session = (people == who) & (days == visit)
            for c in range(3):
                mask = session & (y == c)
                assert mask.any()
                by_class[:, i, v, :, c] = 100*(matrix[:, :, mask] == c).mean(-1)
    cells = by_class.mean(-1)
    for a, arm in enumerate(ARMS):
        for j, seed in enumerate(SEEDS):
            measured = score(y, matrix[a, j], meta, np.ones(ROWS, dtype=bool))
            assert measured["n_cells"] == 18, (arm, seed)
            np.testing.assert_allclose(np.sort(np.ravel(cells[a, :, :, j])),
                                       np.sort(100*np.asarray(measured["cell_scores"])), atol=1e-10, rtol=0)
            np.testing.assert_allclose(cells[a, :, :, j].mean(),
                                       100*measured["balanced_accuracy_cell_mean"], atol=1e-10, rtol=0)
    return cells, by_class


def main(source):
    matrix, warm, proof, people, days, meta = verify(source)
    bindings = dict(scoring_source=sha("scoring.sha256"), scoring_inputs=sha("score_inputs.sha256"))
    lock = seal("before_labels.json", {**proof, **bindings})
    with np.load(DATA, allow_pickle=False) as f:
        labels = f["y"]
    assert labels.shape == (ROWS,) and np.isin(labels, (0, 1, 2)).all()
    cells, by_class = cells_for(labels, matrix, people, days, meta, proof["roster"])
    p, v, o = boot_draws()
    contrasts = {}
    for name, a, b in (("scalar_vs_stock", 1, 0), ("robust_vs_stock", 2, 0), ("robust_vs_scalar", 2, 1)):
        delta = cells[a]-cells[b]
        ci = interval(delta[p, v, o].mean(axis=(1, 2, 3)))
        by_order = dict(zip(SEEDS, delta.mean(axis=(0, 1)).tolist(), strict=True))
        outcome = ("supported" if ci[0] > 0 and min(by_order.values()) > 0
                   else "refuted" if ci[1] <= 0 else "undecided")
        contrasts[name] = dict(pp=float(delta.mean()), paired80=ci, order_pp=by_order, outcome=outcome,
                               class_pp=(by_class[a]-by_class[b]).mean(axis=(0, 1, 2)).tolist(),
                               people_pp=dict(zip(proof["roster"], delta.mean(axis=(1, 2)).tolist(), strict=True)),
                               decisions=counts(matrix[b], matrix[a], labels))
    rt = {a: timing(warm[0], warm[j]) for j, a in enumerate(ARMS[1:], start=1)}
    accuracy = (contrasts["robust_vs_stock"]["outcome"] == "supported"
                and contrasts["robust_vs_scalar"]["paired80"][0] > 0)
    result = dict(scope="FULL-SYSTEM EXTERNAL reused people and fresh delivery",
                  absolute_pct=dict(zip(ARMS, cells.mean(axis=(1, 2, 3)).tolist(), strict=True)),
                  forecast_pp=dict(robust_vs_stock=.35, robust_vs_scalar=.23, scalar_vs_stock=.12),
                  contrasts=contrasts, a100=rt, accuracy_gate=accuracy,
                  advance=accuracy and rt["robust"]["passes_a100_gate"],
                  lock_sha=lock, activity_differences=proof["activity_differences"], effects=proof["effects"],
                  source=source, **bindings)
    digest = seal("result.json", result)
    print(f"PASS F-C15 exact score {digest}", flush=True)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
