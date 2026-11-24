"""Lane P2 (copy of laneP score_b.py over all arms of final_p): official balanced accuracy of the final stack (hardcb) and the arms of one B1 set on the ten reused held-new people (cells equal), paired people bootstrap.
Runs in the c25d worker environment after final_b.py wrote ORDER_SET.npz (B1_SET as there). Contrasts: every live arm minus hardcb (final stack) and minus stock (cap250)."""
import json

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from final_p import ALL as ARMS, SET, WORK

LIVE = tuple(a for a in ARMS if a != "identity")
ORDERS = ("full0", "within0")
CONTRASTS = {f"{a}_minus_final": (a, "hardcb") for a in LIVE if a != "hardcb"}
CONTRASTS.update({f"{a}_minus_cap250": (a, "stock") for a in LIVE if a != "stock"})
CONTRASTS.update({f"{a}_minus_ctl": (a, "ftctl") for a in LIVE if a in ("ftlite5",)})
CONTRASTS.update({f"{a}_minus_{a}_stk": (a, f"{a}_stk") for a in ("ds", "ds_l5", "ds_all")})  # pseudo-label form: packaged (self) minus stock labels
CONTRASTS.update({"ds_l5_minus_ftlite5": ("ds_l5", "ftlite5"), "ds_all_minus_hcF_l5": ("ds_all", "hcF_l5"), "ds_all_minus_ds_l5": ("ds_all", "ds_l5"),
                  "hcF_l5_minus_ftlite5": ("hcF_l5", "ftlite5"), "hcF_l5_minus_hcF": ("hcF_l5", "hcF"), "ds_minus_ctl": ("ds", "ftctl")})


def load(order):
    receipt = json.loads((WORK / f"{order}_{SET}.json").read_text())
    assert receipt["passed"] and receipt["scope"] == "complete" and not receipt["held_labels_opened"]
    pin(WORK / f"{order}_{SET}.npz", receipt["output"]["sha256"])
    with np.load(WORK / f"{order}_{SET}.npz", allow_pickle=False) as z:
        return {a: z[f"{a}|terminal"].argmax(-1) for a in LIVE}, sha(WORK / f"{order}_{SET}.json")


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
    for order in ORDERS:
        g, receipts[order] = load(order)
        cells[order] = person_cells(g, y, subjects, groups, metric)
    result = dict(scope="FULL-SYSTEM DEV ten reused held-new participants", receipts=receipts, orders={})
    rng = np.random.default_rng(460)
    draw = rng.integers(0, 10, (10000, 10))
    names = sorted(cells["full0"])
    unit = {a: i for i, a in enumerate(LIVE)}
    for tag, orders in (("full0", ("full0",)), ("within0", ("within0",)), ("both", ("full0", "within0"))):
        if not set(orders) <= set(ORDERS):
            continue
        # per person sum of cell scores and counts, orders averaged (paired by construction)
        sums = np.array([[np.mean([cells[o][p][:, k].sum() for o in orders]) for k in range(len(LIVE))] for p in names])
        counts = np.array([cells["full0"][p].shape[0] for p in names])
        point = 100 * sums.sum(0) / counts.sum()
        boot = 100 * sums[draw].sum(1) / counts[draw].sum(1)[:, None]
        eff = {}
        for key, (a, b) in CONTRASTS.items():
            d = boot[:, unit[a]] - boot[:, unit[b]]
            eff[key] = dict(pp=float(point[unit[a]] - point[unit[b]]), paired80=np.percentile(d, (10, 90)).tolist())
        for c in ("ds",):  # interaction = joint - lite5 alone - correction alone (all vs hardcb), same bootstrap draws
            j, l, k, h = (unit[x] for x in (f"{c}_l5", "ftlite5", c, "hardcb"))
            d = boot[:, j] - boot[:, l] - boot[:, k] + boot[:, h]
            eff[f"interaction_{c}"] = dict(pp=float(point[j] - point[l] - point[k] + point[h]), paired80=np.percentile(d, (10, 90)).tolist())
        j, l, f, k, h = (unit[x] for x in ("ds_all", "ftlite5", "hcF", "ds", "hardcb"))  # three-way: joint - three single gains, vs hardcb
        d = boot[:, j] - boot[:, l] - boot[:, f] - boot[:, k] + 2 * boot[:, h]
        eff["interaction_all3"] = dict(pp=float(point[j] - point[l] - point[f] - point[k] + 2 * point[h]), paired80=np.percentile(d, (10, 90)).tolist())
        by_person = {p: dict(zip(LIVE, (100 * sums[i] / counts[i]).tolist())) for i, p in enumerate(names)}
        result["orders"][tag] = dict(absolute_pct=dict(zip(LIVE, point.tolist())), contrasts=eff, people_pct=by_person)
    write_receipt(WORK / f"score_{SET}.json", **result)
    print(json.dumps({t: v["contrasts"] for t, v in result["orders"].items()}, indent=1))


if __name__ == "__main__":
    main()
