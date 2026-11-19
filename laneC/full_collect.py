"""Seal ensemble posteriors from the full D1 fit wave for paired mixer replay."""
import argparse
import json
from pathlib import Path

import numpy as np

from collect import ba, bank
from data import ALL, FULL_ORDERS
from oracle import ORDER_SHA, metadata, require, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("work", type=Path)
    args = p.parse_args()
    d = metadata(args.root / "data/proteus_windows.npz")
    archive = args.root / "runs/shuffle_perms.npz"
    require(sha(archive) == ORDER_SHA, "pinned order archive changed")
    out = {"orders": {}}
    with np.load(archive, allow_pickle=False) as permutations:
        for role, people in (("held_new", ALL), ("returning", ("returning",))):
            folds = {held: [bank(args.work, "B", seed, held, d, permutations, FULL_ORDERS, "full_", True)
                            for seed in range(3)] for held in people}
            rows = np.concatenate([v[0][0] for v in folds.values()])
            idx = np.argsort(rows)
            rows = rows[idx]
            want = dict(index=rows)
            out["orders"][role] = {}
            for order in FULL_ORDERS:
                scores = np.concatenate([np.mean([bank[1][order] for bank in v], axis=0)
                                         for v in folds.values()])[idx]
                role_key = "seen_session" if role == "returning" else "all_loso"
                want[order + "_delivery"] = permutations[f"full|{order[-1]}|{role_key}"]
                want[order] = scores
                acc, recall, cells = ba(d, rows, scores.argmax(1))
                out["orders"][role][order] = dict(ba=acc, recall=recall, cells=cells)
            file = args.work / f"ensemble_{role}.npz"
            np.savez(file, **want)
            out["orders"][role]["bank_sha256"] = sha(file)
    (args.work / "full_standalone.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
