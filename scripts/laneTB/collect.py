"""Collect complete paired Team B fold predictions using the official cell metric."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cleanslate/harness"))
import data as D
from score import collect

ORDERS = (71120, 71121, 71122, 71123)
ARMS = ("base", "nine15", "nine25", "three15", "three25")
FOLDS = [(mode, person) for mode, people in (("new", D.PEOPLE), ("ret", D.RETURNING)) for person in people]
KEYS = ("mode", "order", "person", "session", "interface")


def load_arm(root, arm, data):
    rows_by_fold = {}
    for mode, person in FOLDS:
        path = root / f"{mode}_{person}/{arm}/pred_{mode}_{person}.npz"
        with np.load(path, allow_pickle=False) as part:
            sessions = sorted(set(data["session"][data["person"] == person])) if mode == "new" else ("2", "3")
            plans = {f"{order}_{stream.split('|')[1]}": rows for order in ORDERS
                     for stream, rows in D.plans(data, order, [f"{person}|{s}" for s in sessions]).items()}
            actual = {k.removeprefix("rows_") for k in part.files if k.startswith("rows_")}
            assert actual == set(plans), (arm, mode, person, actual)
            for key, rows in plans.items():
                np.testing.assert_array_equal(part[f"rows_{key}"], rows)
                labels = part[f"labels_{key}"]
                assert labels.shape == rows.shape and np.isin(labels, [0, 1, 2]).all()
            rows_by_fold[mode, person] = str(part["source_sha"]), str(part["weights_sha"])
    cells, preds = {}, {}
    for mode, person in FOLDS:
        part_cells, part_preds = collect(root / f"{mode}_{person}" / arm, data)
        assert part_cells and not (set(preds) & set(part_preds))
        cells.update({tuple(c[k] for k in KEYS): c["ba"] for c in part_cells})
        preds.update(part_preds)
    return cells, preds, rows_by_fold


def table(cells, mode, people):
    groups = [[sorted(k for k in cells if k[:3] == (mode, order, p)) for p in people] for order in ORDERS]
    counts = np.array([len(k) for k in groups[0]])
    assert (counts > 0).all() and all([len(k) for k in g] == counts.tolist() for g in groups)
    return np.array([[np.mean([cells[k] for k in part]) for part in group] for group in groups]), counts


def interval(draws):
    return np.quantile(draws, [0.10, 0.90]).tolist()


def by_cohort(cells, mode, people, orders, rng):
    values, counts = table(cells, mode, people)
    sampled_people = rng.integers(len(people), size=(len(orders), len(people)))
    sampled = values[orders[:, :, None], sampled_people[:, None, :]].mean(1)
    w = counts[sampled_people]
    draws = (sampled * w).sum(1) / w.sum(1)
    per_order = np.average(values, axis=1, weights=counts)
    return dict(mean=float(per_order.mean()), ci80=interval(draws), orders=per_order.tolist(),
                people={p: dict(mean=float(values[:, i].mean()), ci80_order=interval(values[orders, i].mean(1)))
                        for i, p in enumerate(people)}), draws


def timing_report(smoke):
    times, n = smoke["timings"], smoke["windows"] - 1
    hard_pairs = [(h["warm_s"] - s["warm_s"]) / n for h, s in
                  zip(times["safe"]["runs"], times["stock"]["runs"], strict=True)]
    out = dict(device=smoke["device"], hardcb_ms=1000 * np.mean(hard_pairs),
               hardcb_ms_pairs=(1000 * np.array(hard_pairs)).tolist())
    for key, full in (("nine", "nine15"), ("three", "three15")):
        encoder = times[full]["encoder_ms"]
        added_load = times[full]["load_s"] - times["stock"]["load_s"]
        first = [(h["setup_s"] - s["setup_s"]) for h, s in
                 zip(times[full]["runs"], times["stock"]["runs"], strict=True)]
        projected = 2555 + 20083 * (np.mean(hard_pairs) + np.mean(encoder) / 1000)
        out[key] = dict(encoder_ms=float(np.mean(encoder)), encoder_ms_pairs=encoder,
                        projection_s=float(projected), load_delta_s=added_load,
                        first_window_and_copy_delta_s=first)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--data", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = D.load(args.data)
    base, base_pred, receipts = load_arm(args.root, "base", data)
    rng = np.random.default_rng(20261121)
    orders = rng.integers(len(ORDERS), size=(20_000, len(ORDERS)))
    result = dict(orders=ORDERS, fold_sha256={f"{k[0]}_{k[1]}": v for k, v in receipts.items()}, arms={},
                  timing=timing_report(json.loads((args.root / "smoke/smoke.json").read_text())))
    result["baseline_new"], _ = by_cohort(base, "new", D.PEOPLE, orders, rng)
    result["baseline_ret"], _ = by_cohort(base, "ret", D.RETURNING, orders, rng)
    for arm in ARMS[1:]:
        cells, pred, arm_receipts = load_arm(args.root, arm, data)
        assert cells.keys() == base.keys() and pred.keys() == base_pred.keys() and arm_receipts == receipts
        delta = {k: cells[k] - base[k] for k in base}
        # Shared bootstrap samples for all arm contrasts.
        pair_rng = np.random.default_rng(71120)
        new, ndraw = by_cohort(delta, "new", D.PEOPLE, orders, pair_rng)
        ret, rdraw = by_cohort(delta, "ret", D.RETURNING, orders, pair_rng)
        rp = ret["people"]
        proxy = 0.57 * new["mean"] + 0.43 * (0.2 * rp["P04"]["mean"] + 0.8 * rp["P10"]["mean"])
        changed = {mode: sum(np.count_nonzero(pred[k] != base_pred[k]) for k in pred if k.startswith(mode + "|"))
                   for mode in ("new", "ret")}
        result["arms"][arm] = dict(new=new, returning=ret, changed=changed, proxy=proxy,
                                   j=0.5 * (new["mean"] + ret["mean"]), j_ci80=interval(0.5 * (ndraw + rdraw)))
    positives = [a for a in ARMS[1:] if result["arms"][a]["new"]["mean"] > 0]
    result["package_arm"] = max(positives, key=lambda a: result["arms"][a]["proxy"]) if positives else None
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"arms": {a: (v["new"]["mean"], v["new"]["ci80"]) for a, v in result["arms"].items()},
                      "timing": result["timing"], "package_arm": result["package_arm"]}, indent=2))


if __name__ == "__main__":
    main()
