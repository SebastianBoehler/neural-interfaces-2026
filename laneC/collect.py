"""Validate and score the complete D1 matrix, then apply the preregistered gate."""
import argparse
import json
from pathlib import Path

import numpy as np

from data import PEOPLE, TRAIN_ORDERS, fold_rows
from oracle import CAMP, ORDER_SHA, metadata, metric_weights, read_friday, require, sha


def bank(work, arm, seed, held, meta, orders, names=TRAIN_ORDERS, scope="", posterior=False):
    path = work / f"reve_{scope}{arm}_{seed}_{held}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    identity = (receipt["arm"], receipt["seed"], receipt["held"])
    require(identity == (arm, seed, held) and receipt["checked_prefix"]
            and receipt["bank_sha"] == sha(path)
            and receipt["state_sha"] == sha(path.with_suffix(".pt")), f"invalid fit: {path}")
    require((arm != "A") == (receipt["external_state"] is not None), f"external run inactive: {path}")
    person = np.array([s.rsplit("/", 1)[-1] for s in meta["subject"]])
    train, query = fold_rows(person, meta["session"], held)
    np.testing.assert_array_equal([len(train), len(query)], [receipt["train_rows"], receipt["query_rows"]])
    role = "seen_session" if held == "returning" else "all_loso"
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(train, z["train"])
        np.testing.assert_array_equal(query, z["index"])
        for order in names:
            ref = orders[f"{order[:-1]}|{order[-1]}|{role}"]
            np.testing.assert_array_equal(z[f"{order}_delivery"], ref[np.isin(ref, query)])
            scores = z[order]
            require(scores.shape == (len(query), 3) and (scores >= 0).all(), f"bad posterior: {path}")
            np.testing.assert_allclose(scores.sum(1), 1, atol=1e-6, rtol=0)
        return query, {o: z[o] if posterior else z[o].argmax(1) for o in names}, receipt["job"]


def ba(meta, rows, pred):
    mask = np.isin(np.arange(len(meta["y"])), rows)
    w, cells = metric_weights(meta, mask)
    y = meta["y"][rows]
    recalls = [float(300 * (w[rows] @ ((y == label) & (pred == label)))) for label in range(3)]
    return float(100 * (w[rows] @ (pred == y))), recalls, cells


def collect(root, work):
    meta = metadata(root / "data/proteus_windows.npz")
    order_archive = root / "runs/shuffle_perms.npz"
    require(sha(order_archive) == ORDER_SHA, "order manifest differs")
    d, output = {}, {}
    with np.load(order_archive, allow_pickle=False) as permutations:
        old, pins = read_friday(work.parent, {"fc20_outer_v1": ()}, "full0",
                               permutations["full|0|all_loso"])
        for seed in (0, 1):
            for arm in "ABC":
                values = [bank(work, arm, seed, p, meta, permutations) for p in PEOPLE]
                rows = np.concatenate([v[0] for v in values])
                by_order = {}
                for o in TRAIN_ORDERS:
                    y = meta["y"][rows]
                    prediction = np.concatenate([v[1][o] for v in values])
                    score, recall, cells = ba(meta, rows, prediction)
                    phi = None
                    if o == "full0":
                        err = old["stock"].argmax(1)[rows] != y
                        phi = float(np.corrcoef(err, prediction != y)[0, 1])
                        require(np.isfinite(phi), "undefined error correlation")
                    by_order[o] = dict(ba=score, recall=recall, cells=cells, stack_error_phi=phi,
                                       people={p: ba(meta, v[0], v[1][o])[0] for p, v in zip(PEOPLE, values)})
                d[arm, seed] = by_order
                output[f"{arm}_{seed}"] = dict(orders=by_order, jobs=[v[2] for v in values])
        rows, scores, ret_id = bank(work, "B", 0, "returning", meta, permutations)
    out = {
        "new": output, "returning": {o: {p: ba(meta, rows[p_mask], pred[p_mask])[0]
                       for p in ("P04", "P10") if (p_mask := np.char.endswith(meta["subject"][rows], p)).any()}
                       for o, pred in scores.items()}, "returning_job": ret_id, "stock_pins": pins,
    }
    diff = [d["B", seed]["full0"]["ba"] - d["A", seed]["full0"]["ba"] for seed in (0, 1)]
    added = [d["C", seed]["full0"]["ba"] - d["B", seed]["full0"]["ba"] for seed in (0, 1)]
    corr = [d["B", seed]["full0"]["stack_error_phi"] for seed in (0, 1)]
    people_diffs = np.array([np.mean([d["B", s]["full0"]["people"][p]
                                    - d["A", s]["full0"]["people"][p] for s in (0, 1)]) for p in PEOPLE])
    rng = np.random.default_rng(20261119)
    draw = rng.integers(len(PEOPLE), size=(10000, len(PEOPLE)))
    cell_counts = np.array([metric_weights(meta, np.char.endswith(meta["subject"], p))[1] for p in PEOPLE])
    interval = np.quantile(np.average(people_diffs[draw], weights=cell_counts[draw], axis=1), [.1, .9])
    decision = "launch" if np.mean(diff) >= 4 and max(corr) <= .5 else "stop" if max(diff) < 2 else "undecided"
    out.update(B_minus_A_pp=diff, C_minus_B_pp=added, B_error_phi=corr, paired_people_80=interval.tolist(),
               decision=decision)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("work", type=Path)
    a = p.parse_args()
    result = collect(a.root, a.work)
    text = json.dumps(result, indent=2) + "\n"
    (a.work / "smoke_result.json").write_text(text)
    print(text, flush=True)
