"""Select exactly one timed Team B arm or confirm the frozen selection."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import collect as C
from grow_sources import ARMS, CONFIRM_ORDERS, REFERENCE, SELECT_ORDERS
from verify import digest


def aggregate(run, phase, arms, orders, data):
    cells, receipts = {}, {}
    C.ORDERS = tuple(orders)
    with TemporaryDirectory(dir=run) as directory:
        root = Path(directory)
        for arm in arms:
            run_folds = {}
            for mode, held in C.FOLDS:
                fold = f"{mode}_{held}"
                merged = {}
                identity = None
                for first in orders[::2]:
                    path = run / phase / fold / arm / f"pred_{mode}_{held}_{first}.npz"
                    meta = json.loads(path.with_suffix(".json").read_text())
                    meta.pop("load_s")
                    if identity is None:
                        identity = meta
                    assert identity == meta, (phase, arm, fold)
                    with np.load(path, allow_pickle=False) as item:
                        assert str(item["source_sha"]) == meta["source_sha"]
                        assert str(item["weights_sha"]) == meta["bank_sha"]
                        for key in item.files:
                            if key.startswith(("rows_", "labels_")):
                                assert key not in merged, (path, key)
                                merged[key] = item[key]
                target = root / fold / arm
                target.mkdir(parents=True)
                np.savez(target / f"pred_{mode}_{held}.npz", source_sha=identity["source_sha"],
                         weights_sha=identity["bank_sha"], **merged)
                run_folds[fold] = identity
            cells[arm] = C.load_arm(root, arm, data)[0]
            assert cells[arm].keys() == cells["base"].keys()
            for order in orders:
                assert sum(k[:2] == ("new", order) for k in cells[arm]) == 28
                assert sum(k[:2] == ("ret", order) for k in cells[arm]) == 8
            receipts[arm] = run_folds
    return cells, receipts


def cohort(delta, mode, people, rng):
    values, counts = C.table(delta, mode, people)
    num_orders, num_people = values.shape
    picked = rng.integers(num_people, size=(20_000, num_people))
    order = rng.integers(num_orders, size=(20_000, num_people, num_orders))
    w = counts[picked]
    draw = (values[order, picked[:, :, None]].mean(2) * w).sum(1) / w.sum(1)
    per_order = np.average(values, axis=1, weights=counts)
    by_person = dict(zip(people, values.mean(0).tolist(), strict=True))
    return dict(mean=float(per_order.mean()), ci80=C.interval(draw), orders=per_order.tolist(),
                by_person=by_person), draw


def summarize(cells, base=None):
    delta = {k: v - base[k] for k, v in cells.items()} if base is not None else cells
    rng = np.random.default_rng(71140)
    new, nd = cohort(delta, "new", C.D.PEOPLE, rng)
    ret, rd = cohort(delta, "ret", C.D.RETURNING, rng)
    rp = ret["by_person"]
    return dict(new=new, returning=ret, j=(new["mean"] + ret["mean"]) / 2, j_ci80=C.interval((nd + rd) / 2),
                proxy_57_43_unverified=0.57 * new["mean"] + 0.43 * (0.2 * rp["P04"] + 0.8 * rp["P10"]))


def timing(run):
    measurements = json.loads((run / "timing.json").read_text())
    assert measurements["device"].startswith("NVIDIA A100")
    pairs = measurements["pairs"]
    assert pairs.keys() == ARMS.keys()
    output = {}
    for arm, samples in pairs.items():
        assert len(samples) == 2 and all(ref["windows"] == job["windows"] > 180 for ref, job in samples)
        ms = [1000 * (job["warm_s"] - ref["warm_s"]) / (ref["windows"] - 1) for ref, job in samples]
        setup = [job["setup_s"] - ref["setup_s"] for ref, job in samples]
        load = [job["load_s"] - ref["load_s"] for ref, job in samples]
        projection = 2943 + 20083 * np.mean(ms) / 1000 + 106 * max(0, np.mean(setup)) + max(0, np.mean(load))
        output[arm] = dict(device=measurements["device"], delta_ms=ms, setup_delta_s=setup, load_delta_s=load,
                           projection_s=2943.0 if arm == REFERENCE else float(projection), pairs=samples)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("select", "confirm"))
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    a = p.parse_args()
    selection = a.run / "select.json"
    if a.phase == "select":
        orders, arms = SELECT_ORDERS, tuple(ARMS)
        previous = None
    else:
        previous = json.loads(selection.read_text())
        orders = CONFIRM_ORDERS
        arms = tuple(dict.fromkeys(("base", REFERENCE, previous["chosen"])))
    cells, receipts = aggregate(a.run, a.phase, arms, orders, C.D.load(a.data))
    contrasts = {arm: summarize(cells[arm], cells["base"]) for arm in arms if arm != "base"}
    times = timing(a.run)
    result = dict(phase=a.phase, orders=list(orders), against_base=contrasts, timing=times,
                  absolute={arm: summarize(cell) for arm, cell in cells.items()},
                  receipts=receipts, data_sha=digest(a.data))
    if previous is None:
        feasible = [arm for arm in contrasts if times[arm]["projection_s"] <= 3600]
        if not feasible:
            raise ValueError("no candidate meets the A100 proxy budget")
        result["chosen"] = min(feasible, key=lambda arm: (-contrasts[arm]["j"], times[arm]["projection_s"], arm))
    else:
        result["chosen"] = previous["chosen"]
        assert previous["data_sha"] == result["data_sha"]
        assert all(receipts[arm] == previous["receipts"][arm] for arm in arms)
        result["selection_sha"] = digest(selection)
        result["against_nine25"] = summarize(cells[result["chosen"]], cells[REFERENCE])
    target = a.run / f"{a.phase}.json"
    with TemporaryDirectory(dir=a.run) as directory:
        temp = Path(directory) / target.name
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.link(temp, target)
    print(a.phase, result["chosen"], target, digest(target), flush=True)


if __name__ == "__main__":
    main()
