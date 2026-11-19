"""Replay the Lane B terminal changes on the authenticated corrected Friday LOSO instrument."""

import json
from pathlib import Path
import sys

import numpy as np

from collect import NAMES, balanced_accuracy, paired
from make import digest
from native import bitwise, check_stage, fresh_tail
from smoke import PUBLIC_SHA


PINS = {
    "full0": ("full0_common_terminal_v1", "d99fab293f89a9059af7ab35488a22aa3a88f151887339ffaa652efb415ce6a8"),
    "within0": ("within0_terminal_shared_v1", "e7d1ec712824f33d8e8237844e9f87d6509e608170beb294316a9f9bb571d59d"),
}
PERMUTATION_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"


def predict(order, package, data, archive, perm, output, smoke):
    if digest(data) != PUBLIC_SHA or digest(perm) != PERMUTATION_SHA:
        raise ValueError("Public data or Friday order archive changed")
    parent, expected = PINS[order]
    bank = archive / parent / f"matched_{order}.npz"
    if digest(bank) != expected:
        raise ValueError("Corrected Friday bank changed")
    with np.load(bank, allow_pickle=False) as z:
        raw, filtered, terminal = (z[f"corrected|cap250_{k}"] for k in ("raw", "filtered", "terminal"))
        delivery = z["delivery"]
        bitwise(terminal.argmax(1), z["corrected|cap250"], "corrected Friday stock")
    with np.load(perm, allow_pickle=False) as z:
        np.testing.assert_array_equal(delivery, z[f"{order[:-1]}|0|all_loso"])
    with np.load(data, allow_pickle=False) as z:
        x = z["X"].astype(np.float64)
        visit = np.char.add(np.char.add(z["subject"], "|"), z["session"])
    groups = [delivery[visit[delivery] == v] for v in dict.fromkeys(visit[delivery])]
    np.testing.assert_array_equal(np.concatenate(groups), delivery)
    if max(map(len, groups)) >= 250 or len(groups) != 14:
        raise ValueError("Corrected Friday cohort or B4 no-op assumption changed")
    actors = {arm: fresh_tail(package / f"{arm}.zip")[2] for arm in ("off", "B1", "B2", "B3", "B4", "B5")}
    result = {name: np.full(len(x), -1, np.int8) for name in NAMES}
    proof = {name: [] for name in actors}
    for rows in groups[:1] if smoke else groups:
        result["stock"][rows] = terminal[rows].argmax(1)
        for name, csp in actors.items():
            csp.reset_state()
            incoming = raw if name in ("B1", "B5") else filtered
            scores = np.stack([csp.step(x[r], raw[r], incoming[r], False) for r in rows])
            check_stage(csp)
            if name in ("off", "B2", "B4"):
                bitwise(scores, terminal[rows], f"{order} {visit[rows[0]]} {name}")
            if name != "off":
                result[name][rows] = scores.argmax(1)
            proof[name].append(dict(refits=csp.refits, fusions=csp.fused, single_class=csp.nfail))
    if not smoke:
        if any(not np.isin(p, [0, 1, 2]).all() for p in result.values()):
            raise ValueError("Incomplete held-new predictions")
        with output.open("xb") as fp:
            np.savez_compressed(fp, **result, order=order, delivery=delivery, proof=json.dumps(proof))
    print(f"PASS {order} {'smoke' if smoke else 'bank'}", flush=True)


def collect(data, directory):
    if digest(data) != PUBLIC_SHA:
        raise ValueError("Public data changed")
    predictions = {}
    for order in PINS:
        with np.load(directory / f"{order}.npz", allow_pickle=False) as z:
            bank = np.column_stack([z[n] for n in NAMES])
            if bank.shape != (2655, 6) or not np.isin(bank, [0, 1, 2]).all() or z["order"].item() != order:
                raise ValueError(f"Invalid held predictions for {order}")
            predictions[order] = (bank, json.loads(z["proof"].item()))
    with np.load(data, allow_pickle=False) as z:
        y, groups = z["y"], np.stack([z[k] for k in ("subject", "session", "interface")], axis=1)
    cells, inverse = np.unique(groups, axis=0, return_inverse=True)
    if len(cells) != 28 or len(set(cells[:, 0])) != 10:
        raise ValueError("Unexpected public metric cells")
    result = {}
    for order, (bank, proof) in predictions.items():
        ba = np.array([[100 * balanced_accuracy(y[inverse == i], p[inverse == i])
                        for p in bank.T] for i in range(len(cells))])
        result[order] = dict(contrasts=paired(ba[None], cells[:, 0], 1), work=proof)
    with (directory / "held.json").open("x") as fp:
        json.dump(result, fp, allow_nan=False, indent=2)
        fp.write("\n")


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "collect":
        collect(*map(Path, args))
    else:
        order, *paths = args
        predict(order, *map(Path, paths), smoke=mode == "smoke")
