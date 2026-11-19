"""Per-order returning cases for the Lane A dose sweep: friday/ea_returning/native_inner/inputs.py for the eight pinned
orders instead of full0. Runs inside the D-EArN release's import path (assets, inputs); see dose_worker.sbatch."""
import re

import numpy as np

import assets as A
import inputs as I

ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")  # = returning_pair_cpu1 cells
VISITS = ("P04|2", "P04|3", "P10|2", "P10|3")


def perm_key(order):
    kind, k = re.fullmatch(r"(within|full|runs)(\d)", order).groups()
    return f"{kind}|{k}|seen_session"


def order_perm(order):
    with np.load(A.OWNER / "runs/shuffle_perms.npz", allow_pickle=False) as f:
        return f[perm_key(order)]


def reference(who, held, order):
    """Native source-fit view/core1 reference of the held rows under this order (authenticated by proof.json)."""
    view = A.ROOT / (A.BASE + "view_bank1") / (who + "_s23")
    proof, path = A.meta(view / "proof.json"), view / f"{order}.npz"
    if proof["files"][f"{order}.npz"] != A.sha(path) or not any(v["order"] == order for v in proof["checks"]):
        raise ValueError(f"view bank {order} not authenticated")
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["index"], held)
        return {k: z[k] for k in ("proba", "view256")}, A.sha(path)


def zero_cell(who, order, fit):
    root = I.PRIOR / f"cell_{(0 if who == 'P04' else 8) + ORDERS.index(order)}"
    proof = A.meta(root / "proof.json")
    if (proof["fold"] != fit["fold"] or proof["order"] != order or proof["query_labels_read"]
            or proof["accuracy_computed"] or proof["bundle_manifest_sha256"] != I.BUNDLE_SHA
            or proof["source_sha256"] != I.PRIOR_SHA or proof["predictions_sha256"] != A.sha(root / "predictions.npz")):
        raise ValueError(f"native zero cell differs {who} {order}")
    return root


def ea_means(order, perm):
    idx, k, values = np.sort(perm), None, {}
    for slot, family, seeds in ((1, "fbcnet_ea_tf", 5), (3, "shallow_ea", 6)):
        with np.load(I.EA / f"{family}_seen_session_orders.npz", allow_pickle=False) as bank:
            k = bank["orders"].tolist().index(order)
            np.testing.assert_array_equal(bank["index"], idx)
            np.testing.assert_array_equal(bank["delivery"][k], perm)
            np.testing.assert_array_equal(bank["seeds"], np.arange(seeds))
            p = bank["proba"][:, k].mean(0)
        if p.shape != (762, 3) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=2e-6):
            raise ValueError("invalid causal EA mean")
        values[slot] = p
    return values


def load(who, order):
    """-> (package dir, fit, route names, native weights, channels, {day: case}) for the fold of ``who`` and ``order``."""
    data = A.load_rows()
    x, people, days = data[0], data[1], data[2]
    path, fit, names, weights, channels, _ = I.load(who)  # full0-authenticated fold bundle, fit and channel list
    perm = order_perm(order)
    fit = dict(fit)
    fit["reference"], fit["view_sha"] = reference(who, fit["held"], order)
    root, means, cases = zero_cell(who, order, fit), ea_means(order, perm), {}
    for day in ("2", "3"):
        rows = perm[(people[perm] == f"Dreyer2026Proteus/{who}") & (days[perm] == day)]
        loc, pos = np.searchsorted(fit["held"], rows), np.searchsorted(np.sort(perm), rows)
        np.testing.assert_array_equal(fit["held"][loc], rows)
        with np.load(root / "predictions.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["index"], fit["held"])
            cases[day] = dict(rows=rows, x=x[rows], score=z["base_scores"][loc], labels=z["base_labels"][loc],
                              u=fit["reference"]["view256"][loc], core=fit["reference"]["proba"][loc],
                              ea={"causal": {s: means[s][pos] for s in I.GROUPS}})
    return path, fit, names, weights, channels, cases
