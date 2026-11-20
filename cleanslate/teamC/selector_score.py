"""Score frozen selector decisions with the official cell metric."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from selector import RULES, SEED
from selector_stream import ARMS, DATA_SHA, ORDERS, digest, harness

NAMES = ("stock", "oracle", "oracle_gated", *[f"{arm}_{rule}" for arm in ARMS
                            for rule in ("all", *RULES, "ungated")])
COHORTS = {"new": harness.PEOPLE, "ret": harness.RETURNING}
EXPOSURE = 20_083
CAP_SECONDS = 2_555


def score_fold(d, bank, mode, person, cells, counts, timing):
    for order in ORDERS:
        pred = {name: np.full(len(d["y"]), -1, dtype=np.int8) for name in NAMES}
        mask = np.zeros(len(d["y"]), dtype=bool)
        for visit in sorted(set(d["session"][d["person"] == person])):
            if mode == "ret" and visit == "1":
                continue
            suffix = f"{order}_{visit}"
            rows = bank[f"rows_{suffix}"]
            y = d["y"][rows]
            cap = bank[f"cap_{suffix}"]
            mask[rows] = True
            pred["stock"][rows] = cap
            counts["stock", mode] += [len(y), 0, np.count_nonzero(cap == y), 0]
            for arm in ARMS:
                key = f"{arm}_{suffix}"
                c = bank[f"team_{key}"]
                active = bank[f"active_{key}"]
                if arm == "A2":
                    repairs = (c == y) & (cap != y)
                    pred["oracle"][rows] = np.where(repairs, c, cap)
                    pred["oracle_gated"][rows] = np.where(active & repairs, c, cap)
                    counts["oracle", mode] += [len(y), np.count_nonzero(c != cap), repairs.sum(), 0]
                pred[f"{arm}_all"][rows] = c
                pred[f"{arm}_ungated"][rows] = bank[f"ungated_{key}"]
                for rule, p in zip(RULES, bank[f"chosen_{key}"].T, strict=True):
                    name = f"{arm}_{rule}"
                    pred[name][rows] = p
                    counts[name, mode] += [len(y), np.count_nonzero(p != cap),
                                           np.sum((p == y) & (cap != y)), np.sum((p != y) & (cap == y))]
                setup = float(bank[f"setup_{key}"])
                ctime, stime = bank[f"team_secs_{key}"], bank[f"selector_secs_{key}"]
                timing[arm, mode] += [len(rows), active.sum(), setup + ctime.sum(), stime.sum(),
                                      setup + ctime[active].sum(), stime[active].sum()]
        for name, p in pred.items():
            key = name, mode, person, order
            cells[key] = [r["ba"] for r in harness.cell_table(d, p, mask, order, mode)]


def estimates(cells, mode, people, orders):
    total = np.array([[sum(cells[name, mode, p, o]) for o in ORDERS for p in COHORTS[mode]]
                      for name in NAMES]).reshape(len(NAMES), len(ORDERS), len(COHORTS[mode]))
    n = np.array([len(cells["stock", mode, p, ORDERS[0]]) for p in COHORTS[mode]])
    denom = n[people].sum(axis=-1)
    samples = total[:, orders[:, :, None], people[:, None, :]].sum(axis=-1).mean(axis=-1) / denom
    return {
        name: {
            "value": float(total[i].sum() / (len(ORDERS) * n.sum())),
            "per_order": (total[i].sum(axis=-1) / n.sum()).tolist(),
            "per_person": dict(zip(COHORTS[mode], (total[i].mean(axis=0) / n).tolist(), strict=True)),
            "paired_interval": np.quantile(samples[i] - samples[0], [0.1, 0.9]).tolist(),
            **({"vs_shuffle": np.quantile(samples[i] - samples[NAMES.index(name.replace("nearest", "shuffled"))],
                                          [0.1, 0.9]).tolist()} if name.endswith("_nearest") else {}),
        } for i, name in enumerate(NAMES)
    }


def project(timing):
    result = {}
    for arm in ARMS:
        t = {mode: timing[arm, mode][1:] / timing[arm, mode][0] for mode in COHORTS}
        mix = 0.57 * t["new"] + 0.43 * t["ret"]
        result[arm] = {
            "new_active_fraction": t["new"][0], "ret_active_fraction": t["ret"][0],
            "new_team_ms": 1000 * t["new"][1], "ret_team_ms": 1000 * t["ret"][1],
            "mix_full_s": CAP_SECONDS + EXPOSURE * (mix[1] + mix[2]),
            "mix_pruned_s": CAP_SECONDS + EXPOSURE * (mix[3] + mix[4]),
            "mix_pruned_team_s": EXPOSURE * mix[3], "mix_pruned_selector_s": EXPOSURE * mix[4],
            "new_pruned_s": CAP_SECONDS + EXPOSURE * (t["new"][3] + t["new"][4]),
            "rtx_over_a100_max": (3600 - CAP_SECONDS - EXPOSURE * mix[4]) / (EXPOSURE * mix[3]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.data) != DATA_SHA:
        raise ValueError("unexpected PROTEUS export")
    if "OK" not in (args.work / "replay.log").read_text().splitlines():
        raise ValueError("historical replay did not pass")
    if sum(line.startswith("OK ") for line in (args.work / "blind_smoke.log").read_text().splitlines()) != 2:
        raise ValueError("both smoke folds must pass")
    d = harness.load(args.data)
    cells, hashes = {}, {}
    counts = defaultdict(lambda: np.zeros(4, dtype=int))
    timing = defaultdict(lambda: np.zeros(6))
    for mode, people in COHORTS.items():
        for person in people:
            path = args.work / "blind_full" / f"{mode}_{person}.npz"
            with np.load(path, allow_pickle=False) as bank:
                score_fold(d, bank, mode, person, cells, counts, timing)
                hashes[f"{mode}|{person}"] = dict(blind=digest(path), sources=bank["source_sha"].tolist(),
                                                 weights=bank["package_sha"].tolist(), gpu=str(bank["gpu"]),
                                                 recognition=json.loads(str(bank["recognition"])))
    rng = np.random.default_rng(SEED)
    boot_orders = rng.integers(len(ORDERS), size=(50_000, len(ORDERS)))
    scores = {mode: estimates(cells, mode, rng.integers(len(people), size=(len(boot_orders), len(people))),
                             boot_orders) for mode, people in COHORTS.items()}
    stock = json.loads((args.work / "stock/scores.json").read_text())
    for mode in COHORTS:
        n = sum(len(cells["stock", mode, p, ORDERS[0]]) for p in COHORTS[mode])
        for order, value in zip(ORDERS, scores[mode]["stock"]["per_order"], strict=True):
            receipt = stock[f"{mode}|{order}"]
            if receipt["n_cells"] != n or not np.isclose(receipt["cell_mean"], value, rtol=0, atol=1e-10):
                raise ValueError("stock cell score differs from the original pinned-order score")
    score = dict(scores=scores, timing=project(timing), sources=hashes,
                 counts={f"{name}|{mode}": values.tolist() for (name, mode), values in counts.items()})
    (args.work / "selector_result.json").write_text(json.dumps(score, indent=2) + "\n")


if __name__ == "__main__":
    main()
