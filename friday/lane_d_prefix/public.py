"""Causal D-c replay from the complete corrected Friday prediction banks."""

import json
from pathlib import Path
import sys

import numpy as np

from balance import PrefixBalance
from banks import DATA, ORDERS, digest, metadata, score, stock_bank, verify_manifest, verify_stock


def replay(terminal):
    state = PrefixBalance()
    return np.array([state.step(p).argmax() for p in terminal])


def main(output):
    verify_manifest()
    subject, visit, cell = metadata()
    sealed = {}
    for order in ORDERS:
        delivery, stock, terminal, _ = stock_bank(order, len(subject))
        trial = np.full_like(stock, -1)
        for session in np.unique(visit):
            rows = delivery[visit[delivery] == session]
            trial[rows] = replay(terminal[rows])
            np.testing.assert_array_equal(stock[rows[:19]], trial[rows[:19]])
            np.testing.assert_array_equal(replay(terminal[rows[:35]]), trial[rows[:35]])
        if (trial < 0).any() or np.array_equal(trial, stock):
            raise ValueError(f"D-c incomplete or inactive on {order}")
        sealed[f"{order}_stock"], sealed[f"{order}_trial"] = stock, trial
    blind = output / "dc_public_blind.npz"
    with blind.open("xb") as handle:
        np.savez_compressed(handle, **sealed)
    proof = digest(blind)
    with np.load(DATA, allow_pickle=False) as z:
        y = z["y"]
    result = {"blind_sha256": proof}
    with np.load(blind, allow_pickle=False) as bank:
        for order in ORDERS:
            result[order] = score(y, bank[f"{order}_stock"][None], bank[f"{order}_trial"][None], subject, cell)
            verify_stock(order, result[order]["stock_pct"])
    with (output / "dc_public.json").open("x") as fp:
        json.dump(result, fp, indent=2)
        fp.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
