"""Lane P (copy of laneB1 score_b.py over all arms of final_p): official balanced accuracy of the final stack (hardcb) and the arms of one B1 set on the ten reused held-new people (cells equal), paired people bootstrap.
Runs in the c25d worker environment after final_b.py wrote ORDER_SET.npz (B1_SET as there). Contrasts: every live arm minus hardcb (final stack) and minus stock (cap250)."""
import json
from pathlib import Path

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from final_w import ALL as ARMS, SET, WORK

ORDERS = ("full0", "within0")
SETS = ("a", "b")  # LANEW_SET job files {order}_W{set}.npz; arms are merged by name (base arms of set a are the reference, set b must reproduce them)


def load(order):
    out, rec = {}, {}
    for st in SETS:
        receipt = json.loads((WORK / f"{order}_{SET}{st}.json").read_text())
        assert receipt["passed"] and receipt["scope"] == "complete" and not receipt["held_labels_opened"]
        pin(WORK / f"{order}_{SET}{st}.npz", receipt["output"]["sha256"])
        with np.load(WORK / f"{order}_{SET}{st}.npz", allow_pickle=False) as z:
            for k in z.files:
                if k.endswith("|terminal"):
                    a = k[:-9]
                    if a in out:
                        np.testing.assert_array_equal(out[a], z[k].argmax(-1), err_msg=f"{a} differs between job sets")  # shared base arms are deterministic
                    out[a] = z[k].argmax(-1)
        rec[st] = sha(WORK / f"{order}_{SET}{st}.json")
    with np.load(Path("/mnt/lustre/work/gehler/gfh527/ni26_laneP_20261120/chain") / f"{order}_P.npz", allow_pickle=False) as z:  # zero-change guards vs lane P
        for mine, theirs in (("stock", "stock"), ("hardcb", "hardcb"), ("hcF_l5", "hcF_l5"), ("hc_s0disp", "ds"), ("da_s0disp", "ds_all")):
            np.testing.assert_array_equal(out[mine], z[f"{theirs}|terminal"].argmax(-1), err_msg=f"{mine} != lane P {theirs}")
    return out, rec


def main():
    pin(METRIC, METRIC_HASH); pin(FEAT, HASHES[FEAT])
    metric = start_module("final_official_metric", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        y, subjects, sessions, interface = (z[k] for k in ("y", "subject", "session", "interface"))
    groups = np.stack((subjects, sessions, interface), axis=1)
    result = dict(scope="FULL-SYSTEM DEV ten reused held-new participants, equal-cell BA per person (selection/intervals in laneW/analyze.py)", receipts={}, by_person={})
    for order in ORDERS:
        g, result["receipts"][order] = load(order)
        arms = sorted(g)
        result["by_person"][order] = {}
        for who in sorted(set(subjects)):
            m = subjects == who
            cells = np.column_stack([metric.group_scores(metric.balanced_accuracy, y[m], g[a][m], groups[m]) for a in arms])
            result["by_person"][order][who] = dict(cells=int(cells.shape[0]), **{a: float(100 * cells[:, i].mean()) for i, a in enumerate(arms)})
    write_receipt(WORK / f"score_{SET}.json", **result)
    print("arms", len(arms))


if __name__ == "__main__":
    main()
