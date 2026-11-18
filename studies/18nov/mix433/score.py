"""Shift evaluation weights across observed delivery schemes, interfaces and modes."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "evidence/next_wave_16nov"
spec = importlib.util.spec_from_file_location("lineage", ROOT / "studies/18nov/lineage419/score.py")
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)
PEOPLE = list(lineage.PEOPLE["all_loso"])
SCHEMES = ["within", "full", "runs"]
TASKS = ["AcquisitionBH", "OnlineRawBH", "AcquisitionGraz", "OnlineRawGraz"]


def banks():
    assert (ROOT / "runs/mix433/replay.txt").read_text().splitlines()[-1] == "OK"
    source = json.loads((EVIDENCE / "aux430_result.json").read_text())
    with np.load(ROOT / "data/proteus_windows.npz") as data:
        meta = {k: data[k] for k in ("subject", "session", "interface", "task", "y", "split")}
    assert np.unique(meta["task"]).tolist() == sorted(TASKS)
    bank = []

    def read(protocol, order, rows, perm, labels, cells, owners, old, historical):
        path = ROOT / f"runs/aux430/{order}.npz"
        assert lineage.pin(path, source["three_sha256"][order])
        with np.load(path) as data:
            cap = data["scores"]
        assert np.isfinite(cap).all() and protocol == "all_loso"
        assert np.array_equal(labels, meta["y"][rows])
        assert np.array_equal(owners, np.char.partition(meta["subject"][rows], "/")[:, 2])
        parts = np.stack([meta[k][rows] for k in ("subject", "session", "interface")], axis=1)
        assert len(np.unique(parts, axis=0)) == len(np.unique(cells)) == 28
        mass = np.zeros((len(PEOPLE), len(TASKS), 3))
        for cell in np.unique(cells):
            indices = np.flatnonzero(cells == cell)
            assert len(np.unique(parts[indices], axis=0)) == 1
            person = PEOPLE.index(owners[indices[0]])
            interface = parts[indices[0], 2]
            for cl in range(3):
                held = indices[labels[indices] == cl]
                assert len(held)
                for t, task in enumerate(TASKS):
                    if interface != ("Graz" if task.endswith("Graz") else "BrainHero"):
                        continue
                    subset = held[meta["task"][rows[held]] == task]
                    assert len(subset)
                    mass[person, t] += np.array([len(subset), (old[subset].argmax(1) == cl).sum(),
                                                  (cap[subset].argmax(1) == cl).sum()]) / (3 * len(held))
        bank.append(mass)

    next(lineage.guarded_matrices(observe_order=read))
    assert len(bank) == len(source["orders"]) == 8
    bank = np.asarray(bank)
    assert np.allclose(bank[..., 0].sum(axis=(1, 2)), 28)
    groups = [np.flatnonzero(np.char.startswith(source["orders"], s)) for s in SCHEMES]
    expected = source["three_minus_old"]["official"]
    endpoints = np.mean([bank[g].mean(0).sum(axis=(0, 1)) for g in groups], axis=0)
    np.testing.assert_allclose(100 * endpoints[1:] / endpoints[0], [expected["old"], expected["new"]], atol=1e-10, rtol=0)
    split = {p: str(np.unique(meta["split"][meta["subject"] == f"Dreyer2026Proteus/{p}"]).item()) for p in PEOPLE}
    return bank, groups, endpoints, split


def stress(mass):
    counts = mass[..., 0]
    gain = (100 * (mass[..., 2] - mass[..., 1]) / counts).ravel()
    weights = counts.ravel() / counts.sum()
    order = np.argsort(gain)
    minimum = gain[order[0]]
    baseline = weights @ gain
    budget = 0.2
    worst = baseline
    # An optimal transfer always pays the minimum stratum and drains the largest gains first.
    for j in order[:0:-1]:
        moved = min(budget, weights[j])
        worst -= moved * (gain[j] - minimum)
        budget -= moved
    distance, balance = 0., baseline
    for j in order[:0:-1]:
        if balance <= 0 or gain[j] == minimum:
            break
        moved = min(weights[j], balance / (gain[j] - minimum))
        distance += moved
        balance -= moved * (gain[j] - minimum)
    threshold = None if minimum > 0 else distance
    marginal = 100 * (mass[..., 2] - mass[..., 1]).sum(axis=0) / counts.sum(axis=0)
    return gain, weights, np.r_[marginal, worst, minimum], threshold


def main():
    bank, groups, endpoints, split = banks()
    per_person = np.stack([bank[g].mean(axis=0) for g in groups])
    gain, weights, result, threshold = stress(per_person.sum(axis=1))
    held = {}
    for j, person in enumerate(PEOPLE):
        train = per_person[:, np.arange(len(PEOPLE)) != j].sum(axis=1)
        least = int(stress(train)[0].argmin())
        held[person] = dict(scheme=SCHEMES[least // 4], task=TASKS[least % 4],
                            evaluated_pp=float(stress(per_person[:, j])[0][least]))
    rng = np.random.default_rng(433)
    draws = []
    for _ in range(6000):
        people = rng.integers(len(PEOPLE), size=len(PEOPLE))
        draw = np.stack([bank[rng.choice(g, size=len(g))].mean(axis=0)[people].sum(axis=0) for g in groups])
        draws.append(stress(draw)[2])
    quant = np.percentile(draws, [10, 90], axis=0).T
    report = dict(people=PEOPLE, released_split=split, endpoints_pct=(100 * endpoints[1:] / endpoints[0]).tolist(),
                  schemes=SCHEMES, tasks=TASKS, gain_pp=gain.reshape(3,4).tolist(),
                  baseline_weight=weights.reshape(3,4).tolist(), distance_to_zero=threshold, leave_person_out=held,
                  mean_held_gain_pp=np.mean([v["evaluated_pp"] for v in held.values()]),
                  effects={k: dict(pp=v, interval80=q.tolist()) for k,v,q in zip(TASKS+["worst_tv20","worst_any"],result,quant)})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
