"""Authenticate one causal member bank and seal its seed mean in recording layout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")
SEED_COUNTS = {"fbcnet_ea_tf": 5, "shallow_ea": 6}
PERMUTATION_SHA256 = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
ROW_COUNT = 2655


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pinned_delivery(order_file):
    if sha256(order_file) != PERMUTATION_SHA256:
        raise ValueError("delivery file differs from the pinned historical file")
    with np.load(order_file, allow_pickle=False) as archive:
        return np.stack([archive[f"{name[:-1]}|{name[-1]}|all_loso"] for name in ORDERS])


def seal_mean(member, bank_file, expected_bank_sha, order_file, output):
    bank_sha = sha256(bank_file)
    if bank_sha != expected_bank_sha:
        raise ValueError(f"{member} bank differs from its receipt")
    seeds = np.arange(SEED_COUNTS[member])
    rows = np.arange(ROW_COUNT)
    delivery = pinned_delivery(order_file)
    with np.load(bank_file, allow_pickle=False) as bank:
        if (not np.array_equal(bank["index"], rows) or not np.array_equal(bank["seeds"], seeds)
                or tuple(bank["orders"]) != ORDERS or not np.array_equal(bank["delivery"], delivery)):
            raise ValueError(f"{member} bank differs from canonical rows, seed set, or delivery order")
        probabilities = bank["proba"]
    if probabilities.shape != (len(seeds), len(ORDERS), ROW_COUNT, 3):
        raise ValueError(f"{member} bank is incomplete")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=-1), 1, rtol=0, atol=2e-6)):
        raise ValueError(f"{member} bank has invalid probabilities")

    mean_recording_layout = probabilities.mean(axis=0)
    with output.open("xb") as handle:
        np.savez_compressed(handle, orders=np.array(ORDERS), index=rows, delivery=delivery,
                            proba=mean_recording_layout, member=member,
                            bank_sha256=bank_sha, permutations_sha256=PERMUTATION_SHA256)
    receipt = dict(member=member, bank_sha256=bank_sha, seed_count=len(seeds),
                   orders=list(ORDERS), rows=ROW_COUNT, mean_sha256=sha256(output),
                   layout="recording rows for each delivery", held_labels_opened=False)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("member", choices=SEED_COUNTS)
    parser.add_argument("bank", type=Path)
    parser.add_argument("bank_sha")
    parser.add_argument("orders", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    seal_mean(args.member, args.bank, args.bank_sha, args.orders, args.output)
