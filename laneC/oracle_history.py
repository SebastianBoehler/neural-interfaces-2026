"""Extend the fast oracle with qualified historical foundation and waveform banks."""
import argparse
import json
from pathlib import Path

import numpy as np

from oracle import ORDER_SHA, metadata, metric_weights, read_friday, require, sha, summarize


def foundation(root, d):
    folder = root / "runs/experiments/probs_fm"
    with np.load(root / "data/proteus_windows.npz", allow_pickle=False) as z:
        dev_rows = np.flatnonzero(z["split"] != "test")
    result, pins = {}, {}
    for name, role, keys, rows in (
        ("LaBraM", "dev_loso", "labram_ea", dev_rows),
        ("CBraMod", "all_loso", "cbramod_ea_lr0.0001_e30", np.arange(len(d["y"]))),
    ):
        paths = sorted(folder.glob(f"{keys}-{role}-s*.npz"))
        require(len(paths) == (2 if role == "dev_loso" else 6), "historical bank inventory")
        members = []
        for path in paths:
            pins[str(path.relative_to(root))] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z["index"], np.arange(len(rows)))
                members.append(z["proba"])
        result[name] = np.full((len(d["y"]), 3), np.nan)
        result[name][rows] = np.mean(members, axis=0)
    return result, pins


def matched_reve(root, d):
    p = root / "runs/diag/reve280"
    result = {}
    for form in ("rv1", "rv2"):
        values = np.empty((len(d["y"]), 3))
        for person in sorted(set(d["subject"])):
            rows = np.flatnonzero(d["subject"] == person)
            members = []
            for seed in (0, 1, 2):
                with np.load(p / f"{form}_s{seed}_{person[-3:]}.npz", allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["rows"], rows)
                    members.append(z["proba_within0"])
            values[rows] = np.mean(members, axis=0)
        result[f"REVE_{form}"] = values
    return result


def waveform(root, d, stock, order):
    p = root / "runs/diag/wave379_review"
    forms = ("static", "norm", "memory")
    result = {k: np.full(len(d["y"]), -1) for k in forms}
    for person in sorted(set(d["subject"])):
        rows = np.flatnonzero(d["subject"] == person)
        short = person.rsplit("/", 1)[-1]
        path = p / f"{short}.npz"
        receipt = json.loads(path.with_suffix(".json").read_text())
        require(receipt["held"] == short and short not in receipt["train_people"]
                and receipt["order_sha256"] == ORDER_SHA and receipt["outputs_sha256"] == sha(path),
                f"invalid waveform bank: {path}")
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["index"], rows)
            for key in forms:
                require(np.isin(z[f"{key}|{order}"], (0, 1, 2)).all(), f"invalid labels in {path}")
                result[key][rows] = z[f"{key}|{order}"]
    y = d["y"]
    weights, _ = metric_weights(d, np.ones(len(y), bool))
    miss = stock.argmax(1) != y
    repairs = np.stack([(pred == y) & miss for pred in result.values()])
    return dict(rows=len(y), unrestricted_union_pp=100 * weights @ repairs.any(0),
                members={key: dict(ba=100 * weights @ (pred == y), unrestricted_pp=100 * weights @ ok,
                                   unrestricted_unique_pp=100 * weights @ (ok & (repairs.sum(0) == 1)))
                         for (key, pred), ok in zip(result.items(), repairs)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    root, work = args.root, args.work
    d = metadata(root / "data/proteus_windows.npz")
    archive = root / "runs/shuffle_perms.npz"
    require(sha(archive) == ORDER_SHA, "historical delivery archive changed")
    old, pins = foundation(root, d)
    output = {"source_sha256": pins}
    with np.load(archive, allow_pickle=False) as z:
        for order in ("full0", "within0"):
            stock, _ = read_friday(work, {"fc20_outer_v1": ()}, order, z[f"{order[:-1]}|0|all_loso"])
            reference = stock["stock"]
            pools = {f"recording_{key}": {key: val} for key, val in old.items()}
            pools["recording_overlap"] = old
            if order == "within0":
                pools["REVE_within0"] = matched_reve(root, d)
            output[order] = summarize(d, reference, pools, .60)
            output[order]["waveform_label_only"] = waveform(root, d, reference, order)
    (work / "oracle_history.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
