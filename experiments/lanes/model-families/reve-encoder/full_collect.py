"""Seal both D1 ensemble banks and the source manifest for paired cap250 replay."""
import argparse
import json
from pathlib import Path

import numpy as np

from collect import ba, bank
from data import ALL, EXTERNAL, FULL_ARMS, FULL_ORDERS, FULL_SEEDS
from oracle import ORDER_SHA, metadata, require, sha


def validate_sources(work, arm, n_rows):
    receipts = {}
    for seed in range(FULL_SEEDS):
        ext = work / f"external_{arm}_{seed}.json"
        fit = work / f"reve_full_{arm}_{seed}_all.json"
        source, model = (json.loads(p.read_text()) for p in (ext, fit))
        expected = set(EXTERNAL)
        if arm == "B":
            expected.remove("scherer2015")
        require(source["arm"] == model["arm"] == arm and source["seed"] == model["seed"] == seed
                and set(source["sources"]) == expected, f"stage A source roster changed for {arm}/{seed}")
        require(model["held"] == "all" and model["train_rows"] == n_rows
                and model["query_rows"] == 0 and model["epochs"] == 25, f"bad deployment fold for {arm}/{seed}")
        require(model["state_sha"] == sha(fit.with_suffix(".pt")), f"invalid all-public state {fit}")
        with np.load(fit.with_suffix(".npz"), allow_pickle=False) as z:
            np.testing.assert_array_equal(z["train"], np.arange(n_rows))
            require(not len(z["index"]), "deployment fold must have no query rows")
        receipts[str(seed)] = dict(external=sha(ext), all_public=sha(fit), state_sha=model["state_sha"],
                                   job=model["job"])
    return receipts


def collect_arm(work, arm, d, permutations):
    output = dict(source_receipts=validate_sources(work, arm, len(d["y"])))
    for role, people, protocol in (("held_new", ALL, "all_loso"), ("returning", ("returning",), "seen_session")):
        folds = {held: [bank(work, arm, seed, held, d, permutations, FULL_ORDERS, "full_", True)
                        for seed in range(FULL_SEEDS)] for held in people}
        rows = np.concatenate([v[0][0] for v in folds.values()])
        idx = np.argsort(rows)
        rows = rows[idx]
        require(len(rows) == (len(d["y"]) if protocol == "all_loso" else 762), f"unexpected {role} coverage")
        want = dict(index=rows)
        scores = {}
        for order in FULL_ORDERS:
            members = []
            for seed in range(FULL_SEEDS):
                members.append(np.concatenate([v[seed][1][order] for v in folds.values()])[idx])
            ensemble = np.mean(members, axis=0)
            delivery = permutations[f"full|{order[-1]}|{protocol}"]
            np.testing.assert_array_equal(np.sort(delivery), rows)
            want[order + "_delivery"] = delivery
            want[order] = ensemble
            want[order + "_members"] = np.stack(members)
            acc, recall, cells = ba(d, rows, ensemble.argmax(1))
            scores[order] = dict(ba=acc, recall=recall, cells=cells)
        file = work / f"ensemble_{arm}_{role}.npz"
        np.savez(file, **want)
        output[role] = dict(orders=scores, file=file.name, bank_sha256=sha(file),
                            actor_jobs={k: [row[2] for row in v] for k, v in folds.items()})
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("work", type=Path)
    args = p.parse_args()
    d = metadata(args.root / "data/proteus_windows.npz")
    archive = args.root / "runs/shuffle_perms.npz"
    require(sha(archive) == ORDER_SHA, "pinned order archive changed")
    with np.load(archive, allow_pickle=False) as permutations:
        report = {arm: collect_arm(args.work, arm, d, permutations) for arm in FULL_ARMS}
    (args.work / "full_standalone.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
