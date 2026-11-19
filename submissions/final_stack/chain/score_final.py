"""Official balanced accuracy of the final-stack arms on the ten reused held-new people (cells equal), paired people bootstrap.
Runs in the c25d worker environment after final22.py wrote full0.npz and within0.npz."""
import json

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from final22 import ARMS, WORK

LIVE = tuple(a for a in ARMS if a != "identity")
CONTRASTS = {"hard_minus_stock": ("hard", "stock"), "cb_minus_stock": ("cb", "stock"), "hardcb_minus_stock": ("hardcb", "stock"),
             "hardcb_minus_hard": ("hardcb", "hard"), "hardcb_minus_cb": ("hardcb", "cb")}


def load(order):
    receipt = json.loads((WORK / f"{order}.json").read_text())
    assert receipt["passed"] and receipt["scope"] == "complete" and not receipt["held_labels_opened"]
    pin(WORK / f"{order}.npz", receipt["output"]["sha256"])
    with np.load(WORK / f"{order}.npz", allow_pickle=False) as z:
        return {a: z[f"{a}|terminal"].argmax(-1) for a in LIVE}, sha(WORK / f"{order}.json")


def person_cells(guesses, truth, subjects, groups, metric):
    out = {}
    for who in sorted(set(subjects)):
        m = subjects == who
        out[who] = np.column_stack([metric.group_scores(metric.balanced_accuracy, truth[m], guesses[a][m], groups[m])
                                    for a in LIVE])  # cells x arms
    return out


def main():
    pin(METRIC, METRIC_HASH); pin(FEAT, HASHES[FEAT])
    metric = start_module("final_official_metric", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        y, subjects, sessions, interface = (z[k] for k in ("y", "subject", "session", "interface"))
    groups = np.stack((subjects, sessions, interface), axis=1)
    cells, receipts = {}, {}
    for order in ("full0", "within0"):
        g, receipts[order] = load(order)
        cells[order] = person_cells(g, y, subjects, groups, metric)
    result = dict(scope="FULL-SYSTEM DEV ten reused held-new participants", receipts=receipts, orders={})
    rng = np.random.default_rng(460)
    draw = rng.integers(0, 10, (10000, 10))
    names = sorted(cells["full0"])
    unit = {a: i for i, a in enumerate(LIVE)}
    for tag, orders in (("full0", ("full0",)), ("within0", ("within0",)), ("both", ("full0", "within0"))):
        # per person sum of cell scores and counts, orders averaged (paired by construction)
        sums = np.array([[np.mean([cells[o][p][:, k].sum() for o in orders]) for k in range(len(LIVE))] for p in names])
        counts = np.array([cells["full0"][p].shape[0] for p in names])
        point = 100 * sums.sum(0) / counts.sum()
        boot = 100 * sums[draw].sum(1) / counts[draw].sum(1)[:, None]
        eff = {}
        for key, (a, b) in CONTRASTS.items():
            d = boot[:, unit[a]] - boot[:, unit[b]]
            eff[key] = dict(pp=float(point[unit[a]] - point[unit[b]]), paired80=np.percentile(d, (10, 90)).tolist())
        inter = boot[:, unit["hardcb"]] - boot[:, unit["hard"]] - boot[:, unit["cb"]] + boot[:, unit["stock"]]
        eff["interaction"] = dict(pp=float(point[unit["hardcb"]] - point[unit["hard"]] - point[unit["cb"]] + point[unit["stock"]]),
                                  paired80=np.percentile(inter, (10, 90)).tolist())
        by_person = {p: dict(zip(LIVE, (100 * sums[i] / counts[i]).tolist())) for i, p in enumerate(names)}
        result["orders"][tag] = dict(absolute_pct=dict(zip(LIVE, point.tolist())), contrasts=eff, people_pct=by_person)
    write_receipt(WORK / "score.json", **result)
    print(json.dumps({t: v["contrasts"] for t, v in result["orders"].items()}, indent=1))


if __name__ == "__main__":
    main()
