"""Lane B7 scorer: seed spread of the lane P arms over the seed sets of final_b7.py (official balanced accuracy, ten reused held-new people, orders full0 + within0).
Per set s and arm a: delta vs cap250 (stock), vs the same set's final stack (hardcb_s) and vs the pinned final stack (hardcb of B0). Spread = min/max/SD over sets.
Pooled estimate = mean over the sets of the delta; 80 % interval from 10,000 draws that resample people (10 with replacement) AND the two orders (2 with replacement), paired over sets and arms.
Runs in the c25d worker environment after final_b7.py wrote ORDER_SET.npz for every set (worker.sbatch score)."""
import json

import numpy as np

from bridge_orders import write_receipt
from bridge_owner import pin, sha, start_module
from complete_score import METRIC, METRIC_HASH
from complete_zero import FEAT, HASHES
from final_b7 import SETS, WORK

ORDERS = ("full0", "within0")
ARMS = ("stock", "hardcb", "ftlite5", "hcF_l5", "ds_l5", "ds_all")
TARGETS = ("ds_all", "ds_l5", "hcF_l5", "ftlite5")
POOLS = {"all": tuple(SETS), "joint": ("J0", "J1", "J2"), "single": tuple(k for k in SETS if k not in ("J0", "J1", "J2"))}


def load(order, name):
    receipt = json.loads((WORK / f"{order}_{name}.json").read_text())
    assert receipt["passed"] and receipt["scope"] == "complete" and not receipt["held_labels_opened"] and receipt["arm_set"] == name
    pin(WORK / f"{order}_{name}.npz", receipt["output"]["sha256"])
    with np.load(WORK / f"{order}_{name}.npz", allow_pickle=False) as z:
        return {a: z[f"{a}|terminal"].argmax(-1) for a in ARMS}, sha(WORK / f"{order}_{name}.json")


def main():
    pin(METRIC, METRIC_HASH); pin(FEAT, HASHES[FEAT])
    metric = start_module("final_official_metric", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        y, subjects, sessions, interface = (z[k] for k in ("y", "subject", "session", "interface"))
    groups = np.stack((subjects, sessions, interface), axis=1)
    names = sorted(set(subjects))
    assert len(names) == 10
    sets = list(SETS)
    sums = np.zeros((len(sets), len(ORDERS), len(ARMS), 10))  # sum of cell scores per person
    counts = np.array([len({tuple(g) for g in groups[subjects == p]}) for p in names], float)
    receipts = {}
    for i, name in enumerate(sets):
        for o, order in enumerate(ORDERS):
            g, receipts[f"{order}_{name}"] = load(order, name)
            for j, p in enumerate(names):
                m = subjects == p
                for k, a in enumerate(ARMS):
                    cells = metric.group_scores(metric.balanced_accuracy, y[m], g[a][m], groups[m])
                    assert len(cells) == counts[j]
                    sums[i, o, k, j] = np.sum(cells)
    ix = {a: k for k, a in enumerate(ARMS)}
    sx = {s: i for i, s in enumerate(sets)}

    def ba(weights_p, order_w):  # weights over people (10,), over orders (2,) -> (sets, arms) pooled BA in pp
        return 100 * np.einsum("soap,p,o->sa", sums, weights_p, order_w) / (weights_p @ counts)

    def delta(table, a, ref, s):  # ref: "stock" | "same" | "pinned"
        r = {"stock": table[sx[s], ix["stock"]], "same": table[sx[s], ix["hardcb"]], "pinned": table[sx["B0"], ix["hardcb"]]}[ref]
        return table[sx[s], ix[a]] - r
    point = ba(np.ones(10), np.array([.5, .5]))
    rng = np.random.default_rng(470)
    D = 10000
    pw = np.stack([np.bincount(r, minlength=10) for r in rng.integers(0, 10, (D, 10))]).astype(float)
    ow = np.stack([np.bincount(r, minlength=2) for r in rng.integers(0, 2, (D, 2))]).astype(float) / 2
    boot = np.stack([ba(pw[d], ow[d]) for d in range(D)])  # D x sets x arms
    out = dict(receipts=receipts, scope="FULL-SYSTEM DEV ten reused held-new people, orders full0+within0 (both-order mean)", absolute_pct={f"{s}|{a}": float(point[sx[s], ix[a]]) for s in sets for a in ARMS}, arms={})
    for a in TARGETS:
        rec = {}
        for ref in ("stock", "same", "pinned"):
            per = {s: float(delta(point, a, ref, s)) for s in sets}
            v = np.array(list(per.values()))
            rec[ref] = dict(per_set=per, min=float(v.min()), max=float(v.max()), sd=float(v.std(ddof=1)), range=float(v.max() - v.min()))
            for pool, members in POOLS.items():
                d = np.mean([boot[:, sx[s], ix[a]] - {"stock": boot[:, sx[s], ix["stock"]], "same": boot[:, sx[s], ix["hardcb"]], "pinned": boot[:, sx["B0"], ix["hardcb"]]}[ref] for s in members], axis=0)
                rec[ref][f"pooled_{pool}"] = dict(pp=float(np.mean([per[s] for s in members])), people_orders80=np.percentile(d, (10, 90)).tolist(), frac_positive=float((d > 0).mean()), sets_positive=int(sum(per[s] > 0 for s in members)), n_sets=len(members))
        out["arms"][a] = rec
    write_receipt(WORK / "score_B7.json", **out)
    for a, rec in out["arms"].items():
        for ref, r in rec.items():
            print(a, "vs", ref, f"B0 {r['per_set']['B0']:+.3f} min {r['min']:+.3f} max {r['max']:+.3f} sd {r['sd']:.3f}",
                  {p: (round(r[f"pooled_{p}"]["pp"], 3), [round(x, 3) for x in r[f"pooled_{p}"]["people_orders80"]]) for p in POOLS})


if __name__ == "__main__":
    main()
