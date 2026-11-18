"""Lock and score the complete fresh-order F-C14 bank with the official metric."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from inputs import ACTOR, ARMS, SEEDS, archive_parity, schedule, seal, sha
from metrics import boot_draws, counts, interval, timing
from protocol import score


ROWS = 2130
ARCHIVED = (411, 1411)
DATA = Path("/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/data/scherer2015/windows.npz")
PRIOR = DATA.parents[3] / "ni26_cross411b_20261118"


def read(path):
    return json.loads(path.read_text())


def verified(path):
    digest = sha(path)
    assert path.with_suffix(".sha256").read_text().strip() == digest
    return read(path), digest


def qualified_bank(source):
    assert sha("source.sha256") == source
    smoke, smoke_sha = verified(Path("smoke.json"))
    assert smoke["passed"] and smoke["source"] == source and smoke["actor"] == ACTOR
    assert not smoke["labels_opened"]
    with np.load(DATA, allow_pickle=False) as data:
        people, visits, interfaces = (data[k] for k in ("subject", "session", "interface"))
    stream = np.char.add(np.char.add(people, "|"), visits)
    assert all(len(set(interfaces[stream == k])) == 1 for k in set(stream))
    roster = sorted(set(people))
    deliveries = schedule(people, visits)
    assert roster == read(PRIOR / "result.json")["people"]
    predictions = np.full((4, 2, ROWS), -1, dtype=np.int8)
    warm = np.zeros((4, 9, 2, 2))
    receipt, archives, work = {}, {}, {a: Counter() for a in ARMS}
    effects = {a: Counter() for a in ARMS}
    for i, person in enumerate(roster):
        names = [f"{person}|{day}" for day in (1, 2)]
        ref_path = PRIOR / f"cap_{i}.json"
        normalized = read(ref_path)
        archives[str(ref_path.name)] = sha(ref_path)
        path = Path(f"person_{i}.json")
        bank, digest = verified(path)
        receipt[path.name] = digest
        assert (bank["index"], bank["who"], bank["source"], bank["smoke_sha"], bank["archive"],
                bank["actor"], bank["labels_opened"]) == (i, person, source, smoke_sha,
                    archives[ref_path.name], ACTOR, False)
        anchors = bank["archived_controls"]
        assert set(anchors) == {f"{names[0]}|{s}" for s in ARCHIVED}
        for seed in ARCHIVED:
            key = f"{names[0]}|{seed}"
            assert anchors[key]["rows"] == normalized[key]["rows"] == deliveries[seed][names[0]].tolist()
            archive_parity(anchors[key]["run"], normalized[key])
        assert set(bank["sessions"]) == {f"{name}|{s}" for name in names for s in SEEDS}
        for j, seed in enumerate(SEEDS):
            for d, name in enumerate(names):
                rows = deliveries[seed][name]
                cell = bank["sessions"][f"{name}|{seed}"]
                rotation = (i + 2*j + d) % len(ARMS)
                order = ARMS[rotation:] + ARMS[:rotation]
                assert cell["rows"] == rows.tolist() and cell["order"] == list(order)
                assert (predictions[:, j, rows] == -1).all() and set(cell["arms"]) == set(ARMS)
                stock = cell["arms"]["stock"]
                assert cell["arms"]["cognitive"]["aligned_sha"] == stock["aligned_sha"]
                assert cell["arms"]["directional"]["aligned_sha"] == cell["arms"]["combined"]["aligned_sha"]
                for a, arm in enumerate(ARMS):
                    run = cell["arms"][arm]
                    assert np.isin(run["labels"], (0, 1, 2)).all() and len(run["labels"]) == len(rows)
                    assert np.isfinite(run["seconds"]) and run["seconds"] > 0
                    act = run["activity"]
                    assert set(act) == set(normalized[f"{name}|411"]["activity"])
                    assert act["windows"] == len(rows) and act["errors"] == act["ft_over"] == 0
                    assert act["ft_steps"] == 20*act["ft_refits"]
                    predictions[a, j, rows] = run["labels"]
                    warm[a, i, d, j] = run["seconds"]
                    work[arm].update(k for k in act if act[k] != stock["activity"][k])
                    if run["odds"] is not None:
                        assert arm in ("cognitive", "combined")
                        assert run["odds"]["calls"] == len(rows) and run["odds"]["above_001"] > 0
                        for key in ("calls", "warm_calls", "above_001"):
                            effects[arm][key] += run["odds"][key]
                        for key in ("max_pre", "max_warm"):
                            effects[arm][key] = max(effects[arm][key], run["odds"][key])
                    else:
                        assert arm in ("stock", "directional")
                    if run["history"] is not None:
                        assert arm in ("directional", "combined")
                        assert run["history"]["active_ages"] == cell["arms"]["directional"]["history"]["active_ages"]
                        assert run["history"]["windows"] == len(rows)
                        effects[arm]["capped_updates"] += len(run["history"]["active_ages"])
                    else:
                        assert arm in ("stock", "cognitive")
    assert (predictions >= 0).all() and (warm > 0).all()
    assert work["stock"] == {} and effects["directional"]["capped_updates"] > 0
    for arm in ("cognitive", "combined"):
        assert effects[arm]["calls"] == ROWS*len(SEEDS) and effects[arm]["warm_calls"] > 0
    proof = dict(source=source, smoke_sha=smoke_sha, actor=ACTOR, seeds=SEEDS, people=roster,
                 worker_receipts=receipt, archived_controls=archives, arm_order=ARMS,
                 predictions_sha=hashlib.sha256(predictions.tobytes()).hexdigest(), rows_per_seed=ROWS,
                 activity_differences=work, effects=effects, labels_opened=False)
    meta = dict(stream=stream, interface=interfaces)
    return predictions, warm, proof, meta, people, visits


def score_cells(predictions, labels, meta, people, visits, roster):
    classes = np.empty((4, 9, 2, 2, 3))
    for i, who in enumerate(roster):
        for d, day in enumerate(("1", "2")):
            rows = (people == who) & (visits == day)
            for c in range(3):
                mask = rows & (labels == c)
                assert mask.any()
                classes[:, i, d, :, c] = 100*(predictions[:, :, mask] == c).mean(axis=-1)
    cells = classes.mean(axis=-1)
    for a, arm in enumerate(ARMS):
        for j, seed in enumerate(SEEDS):
            outputs = score(labels, predictions[a, j], meta, np.ones(ROWS, dtype=bool))
            assert outputs["n_cells"] == 18, (arm, seed)
            np.testing.assert_allclose(outputs["balanced_accuracy_cell_mean"]*100,
                                       cells[a, :, :, j].mean(), atol=1e-10, rtol=0)
            np.testing.assert_allclose(np.sort(100*np.asarray(outputs["cell_scores"])),
                                       np.sort(cells[a, :, :, j].flatten()), atol=1e-10, rtol=0)
    return classes, cells


def paired(delta, person, visit, order):
    return dict(pp=float(delta.mean()),
                paired80=interval(delta[person, visit, order].mean(axis=(1, 2, 3))),
                order_pp=dict(zip(SEEDS, delta.mean(axis=(0, 1)).tolist(), strict=True)))


def main(source):
    bank, warm, proof, meta, people, visits = qualified_bank(source)
    bindings = dict(scoring_source=sha("scoring.sha256"), scoring_inputs=sha("score_inputs.sha256"))
    lock = seal("before_labels.json", {**proof, **bindings})
    with np.load(DATA, allow_pickle=False) as f:
        labels = f["y"]
    assert labels.shape == (ROWS,) and np.isin(labels, (0, 1, 2)).all()
    classes, cells = score_cells(bank, labels, meta, people, visits, proof["people"])
    draws = boot_draws()
    summaries = {arm: {**paired(cells[a]-cells[0], *draws),
                       "class_delta_pp": (classes[a]-classes[0]).mean(axis=(0, 1, 2)).tolist(),
                       "people_pp": dict(zip(proof["people"], (cells[a]-cells[0]).mean(axis=(1, 2)).tolist(), strict=True)),
                       "decisions": counts(bank[0], bank[a], labels),
                       "timing": timing(warm[0], warm[a])}
                 for a, arm in enumerate(ARMS[1:], start=1)}
    interaction = paired(cells[3]-cells[1]-cells[2]+cells[0], *draws)
    edges = {arm: paired(cells[3]-cells[a], *draws) for a, arm in enumerate(ARMS[1:3], start=1)}
    for result in summaries.values():
        result["outcome"] = ("supported" if result["paired80"][0] > 0 and min(result["order_pp"].values()) > 0
                             else "refuted" if result["paired80"][1] <= 0 else "undecided")
    accuracy_gate = summaries["combined"]["outcome"] == "supported" and min(v["pp"] for v in edges.values()) >= .10
    outcome = dict(scope="FULL-SYSTEM EXTERNAL reused people and fresh deliveries", seeds=SEEDS,
                   forecast_pp=dict(directional=.10, cognitive=.25, combined=.45, interaction=.10),
                   absolute_pct=dict(zip(ARMS, cells.mean(axis=(1, 2, 3)).tolist(), strict=True)),
                   main=summaries, interaction=interaction, combined_minus_single=edges,
                   accuracy_gate=accuracy_gate, a100_gate=summaries["combined"]["timing"]["passes_a100_gate"],
                   advance=accuracy_gate and summaries["combined"]["timing"]["passes_a100_gate"],
                   valid_comparator="pinned exact stock on each fresh delivery", bootstrap=dict(draws=10000, seed=468, level=.80),
                   lock_sha=lock, source=source, effects=proof["effects"],
                   activity_differences=proof["activity_differences"], **bindings)
    digest = seal("result.json", outcome)
    print(f"PASS F-C14 score {digest}", flush=True)
    print(json.dumps(outcome, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
