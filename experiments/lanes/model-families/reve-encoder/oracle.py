"""Screen authenticated corrected Friday cap250 against archived held-person members."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DATA_SHA = "2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327"
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
CAMP = "ni26_friday_causal_rebuild_20261118"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def metadata(path):
    require(sha(path) == DATA_SHA, "public export hash changed")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("y", "subject", "session", "interface")}


def metric_weights(d, mask):
    y = d["y"]
    cells = np.char.add(np.char.add(d["subject"], "|" + d["session"]), "|" + d["interface"])
    names = np.unique(cells[mask])
    weights = np.zeros(len(y))
    for name in names:
        cell = (cells == name) & mask
        for label in range(3):
            rows = cell & (y == label)
            require(rows.any(), f"incomplete cell {name}, class {label}")
            weights[rows] = 1 / (3 * len(names) * rows.sum())
    np.testing.assert_allclose(weights.sum(), 1)
    return weights, len(names)


def read_friday(work, folders, order, p):
    out, pins = {}, {}
    for folder, arms in folders.items():
        path = work / CAMP / folder / f"{order}.npz"
        receipt = json.loads(path.with_suffix(".json").read_text())
        digest = sha(path)
        require(receipt["passed"] and receipt["stock_bitwise"] and not receipt["held_labels_opened"]
                and receipt["output"]["sha256"] == digest and receipt["order"] == order, f"invalid receipt: {path}")
        pins[folder] = digest
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["index"], np.arange(len(p)))
            np.testing.assert_array_equal(z["delivery"], p)
            np.testing.assert_array_equal(z["stock|terminal"], z["identity|terminal"])
            if "stock" in out:
                np.testing.assert_array_equal(z["stock|terminal"], out["stock"])
            out["stock"] = z["stock|terminal"]
            for key in arms:
                scores = z[f"{key}|terminal"].copy()
                log_rows = (scores < 0).any(1)
                logits = scores[log_rows]
                prob = np.exp(logits - logits.max(1, keepdims=True))
                scores[log_rows] = prob / prob.sum(1, keepdims=True)
                out[f"{folder}/{key}"] = scores
    return out, pins


def recover(prob, y, miss, threshold):
    require(prob.shape == (len(y), 3) and np.isfinite(prob).all() and (prob >= 0).all(), "invalid probabilities")
    np.testing.assert_allclose(prob.sum(1), 1, atol=2e-5)
    right = prob.argmax(1) == y
    confident = prob.max(1) >= threshold
    return right, miss & right & confident, ~miss & ~right & confident, miss & right


def summarize(d, stock, rosters, threshold):
    y = d["y"]
    miss = stock.argmax(1) != y
    weights, n_cells = metric_weights(d, np.ones(len(y), bool))
    count = len(y)
    out = {"stock_ba": 100 * weights @ ~miss, "cells": n_cells, "rows": count, "pools": {}}
    for name, members in rosters.items():
        mask = np.logical_and.reduce([np.isfinite(a).all(1) for a in members.values()])
        w, cells = metric_weights(d, mask)
        panels = {k: recover(a[mask], y[mask], miss[mask], threshold) for k, a in members.items()}
        repair = np.stack([a[1] for a in panels.values()])
        unrestricted = np.stack([a[3] for a in panels.values()])
        union = repair.any(0)
        ww = w[mask]
        total_miss = ww @ miss[mask]
        out["pools"][name] = dict(
            rows=int(mask.sum()), cells=cells, people=sorted(set(d["subject"][mask])),
            union_confident_pp=100 * ww @ union, union_unrestricted_pp=100 * ww @ unrestricted.any(0),
            stock_ba=100 * ww @ ~miss[mask], members={
                k: dict(ba=100 * ww @ right, repaired_windows=int(ok.sum()),
                        repair_pp=100 * ww @ ok, share_of_misses=100 * (ww @ ok) / total_miss,
                        confident_regression_pp=100 * ww @ reg, unrestricted_pp=100 * ww @ any_repair,
                        unique_pp=100 * ww @ (ok & (repair.sum(0) == 1)))
                for k, (right, ok, reg, any_repair) in panels.items()
            })
    return out


def partial(work, order, d):
    a6 = {name: np.full((len(d["y"]), 3), np.nan) for name in ("A6", "J_real", "K")}
    for person in ("P01", "P02", "P03", "P09"):
        path = work / "ni26_joint382_actor_v1_20261118" / f"{person}.npz"
        receipt = json.loads(path.with_suffix(".json").read_text())
        require(receipt["held"] == person and not receipt["held_labels_opened"]
                and person not in receipt["train_people"] and sha(path) == receipt["outputs_sha256"], "A6 mismatch")
        with np.load(path, allow_pickle=False) as z:
            rows = np.flatnonzero(np.char.endswith(d["subject"], person))
            np.testing.assert_array_equal(rows, z["index"])
            for name, values in a6.items():
                values[rows] = z[f"{name}|{order}"]
    path = work / "ni26_finetune330_replay_v1_20261117/query_probabilities.npz"
    require(sha(path) == "2ec25debcc4135fcfe95a9f1168e43fdb20a0825d3c02b29c9954f1e83ac58a8", "ATOM receipt mismatch")
    atom = {}
    with np.load(path, allow_pickle=False) as z:
        rows = np.flatnonzero(np.char.endswith(d["subject"], "P02") | np.char.endswith(d["subject"], "P03"))
        np.testing.assert_array_equal(z["index"], rows)
        for name in ("pretrained_FT", "pretrained_frozen"):
            values = np.full((len(d["y"]), 3), np.nan)
            values[rows] = z[name]
            atom[f"ATOM-L_{name}_stateless"] = values
    return a6, atom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("banks", type=Path)
    p.add_argument("public", type=Path)
    p.add_argument("orders", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    d = metadata(args.public)
    require(sha(args.orders) == ORDER_SHA, "pinned orders changed")
    out = {"confidence": 0.60, "orders": {}}
    np.testing.assert_equal(len(d["y"]), 2655)
    folders = {"fc20_outer_v1": ("hard", "jitter"), "f25c_v1": ("single", "multi")}
    with np.load(args.orders, allow_pickle=False) as perms:
        for order in ("full0", "within0"):
            delivery = perms[f"{order[:-1]}|{order[-1]}|all_loso"]
            old, pins = read_friday(args.banks, folders, order, delivery)
            stock = old.pop("stock")
            a6, atom = partial(args.banks, order, d)
            rosters = {"stack_alternatives": old, "A6": a6, "ATOM": atom, "matched_all": old | a6 | atom}
            out["orders"][order] = summarize(d, stock, rosters, out["confidence"]) | {"input_sha256": pins}
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
