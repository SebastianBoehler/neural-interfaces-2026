"""Lane R1 scorer: official cell-mean balanced accuracy per person and arm for every ORDER_JOB.npz in this directory (+ lane P stock, hardcb, ds_all).
Writes cells.json {order: {arm: {person: [sum of cell BA, cell count]}}}; intervals and nested selection are computed from it by analyse_r1.py."""
import glob
import os
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES

WORK = Path(__file__).resolve().parent
PREF = Path(os.environ.get("R1_PREF", "/mnt/lustre/work/gehler/gfh527/ni26_laneP_20261120/chain"))


def main():
    pin(METRIC, METRIC_HASH); pin(FEAT, HASHES[FEAT])
    metric = start_module("final_official_metric", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        y, subjects, sessions, interface = (z[k] for k in ("y", "subject", "session", "interface"))
    groups = np.stack((subjects, sessions, interface), axis=1)
    people = sorted(set(subjects))
    out, receipts = {}, {}
    for order in ("full0", "within0"):
        arms = {}
        files = [(PREF / f"{order}_P.npz", ("stock", "hardcb", "ds_all"), "")]
        files += [(Path(f), None, Path(f).stem.split("_", 1)[1] + ":") for f in sorted(glob.glob(str(WORK / f"{order}_*.npz")))]
        for f, only, prefix in files:
            with np.load(f, allow_pickle=False) as z:
                for key in z.files:
                    if not key.endswith("|terminal"):
                        continue
                    arm = key[:-9]
                    if only and arm not in only:
                        continue
                    g = z[key].argmax(-1)  # recording layout, as lane P
                    scores = {}
                    for p in people:
                        m = subjects == p
                        scores[p] = metric.group_scores(metric.balanced_accuracy, y[m], g[m], groups[m])
                    arms[prefix + arm] = {p: [float(s.sum()), int(len(s))] for p, s in scores.items()}
            receipts[f"{order}/{f.name}"] = sha(f)
        out[order] = arms
        print(order, len(arms), "arms", flush=True)
    write_receipt(WORK / "cells.json", receipts=receipts, cells=out, scope="FULL-SYSTEM DEV ten reused held-new people")


if __name__ == "__main__":
    main()
