"""Late fusion of REVE/CBraMod banks into cap250 terminal posteriors (registered in evidence/wave_19nov/laneC_member.md)."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from oracle import metadata, metric_weights, recover, require

GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
ORDERS = [f"full{k}" for k in range(8)]
DRAWS = 10000


def cell_ba(y, pred, cells):
    out = []
    for c in cells:
        recall = [(pred[c & (y == k)] == k).mean() for k in range(3)]
        out.append(np.mean(recall))
    return np.array(out)


def posterior(scores):
    """Rows with negative entries are log scores (oracle.read_friday convention); softmax them."""
    scores = scores.astype(np.float64).copy()
    log_rows = (scores < 0).any(1)
    e = np.exp(scores[log_rows] - scores[log_rows].max(1, keepdims=True))
    scores[log_rows] = e / e.sum(1, keepdims=True)
    np.testing.assert_allclose(scores.sum(1), 1, atol=2e-5)
    return scores


def combos(names):
    return [dict(zip(names, g)) for g in itertools.product(GRID, repeat=len(names))]


def fuse(cap, members, weights):
    score = np.log(np.clip(cap, 1e-6, None))
    for name, w in weights.items():
        score = score + w * np.log(np.clip(members[name], 1e-6, None))
    return score.argmax(1)


def select(delta, combo_list, keep):
    """delta[combo, order, person]: choose the combo maximising the mean over `keep` people and all orders."""
    score = delta[:, :, keep].mean((1, 2))
    return int(np.argmax(score))  # first maximum -> smaller weights on ties, w=0 is combo 0


def new_deltas(d, cap, banks, names, combo_list, orders):
    y = d["y"]
    people = sorted(set(d["subject"]))
    cells = {p: [(d["subject"] == p) & (d["session"] == s) & (d["interface"] == i)
                 for s, i in sorted(set(zip(d["session"][d["subject"] == p], d["interface"][d["subject"] == p])))]
             for p in people}
    ncell = np.array([len(cells[p]) for p in people])
    base = np.array([cell_ba(y, cap.argmax(1), cells[p]).mean() for p in people])
    delta = np.zeros((len(combo_list), len(orders), len(people)))
    for oi, order in enumerate(orders):
        members = {n: banks[n][order] for n in names}
        for ci, combo in enumerate(combo_list):
            pred = fuse(cap, members, combo)
            if ci == 0:
                np.testing.assert_array_equal(pred, cap.argmax(1))
            delta[ci, oi] = 100 * (np.array([cell_ba(y, pred, cells[p]).mean() for p in people]) - base)
    return people, ncell, base, delta


def boot(person_delta, weights, rng):
    """person_delta[order, person]; resample people (cell weighted) and orders."""
    n_o, n_p = person_delta.shape
    po = rng.integers(n_o, size=(DRAWS, n_o))
    pp = rng.integers(n_p, size=(DRAWS, n_p))
    vals = person_delta[po[:, :, None], pp[:, None, :]].mean(1)
    return np.average(vals, weights=weights[pp], axis=1)


def interval(x):
    return [round(float(v), 3) for v in np.quantile(x, [0.1, 0.9])]


def phi(w, a, b):
    ma, mb = w @ a, w @ b
    cov = w @ ((a - ma) * (b - mb))
    return float(cov / np.sqrt(w @ (a - ma) ** 2 * (w @ (b - mb) ** 2)))


def screen(d, cap, bank0):
    y = d["y"]
    w, _ = metric_weights(d, np.ones(len(y), bool))
    miss = cap.argmax(1) != y
    out, rep = {}, {}
    for n, m in bank0.items():
        right, ok, reg, any_rep = recover(m, y, miss, 0.60)
        rep[n] = ok
        out[n] = dict(phi=phi(w, miss.astype(float), (m.argmax(1) != y).astype(float)),
                      repaired_confident_share_pct=float(100 * w @ ok / (w @ miss)),
                      unrestricted_repair_share_pct=float(100 * w @ any_rep / (w @ miss)),
                      confident_regression_pp=float(100 * w @ reg), member_ba=float(100 * w @ right))
    if len(bank0) > 1:
        union = np.any(list(rep.values()), axis=0)
        out["union_confident_repair_share_pct"] = float(100 * w @ union / (w @ miss))
    out["cap250_ba"] = float(100 * w @ ~miss)
    return out


def returning(d, ret_rows, ret_cap, banks, names, combo_list, orders, choose):
    y = d["y"][ret_rows]
    subj, ses, ifc = (d[k][ret_rows] for k in ("subject", "session", "interface"))
    keys = sorted(set(zip(subj, ses, ifc)))
    cells = [(subj == s) & (ses == a) & (ifc == b) for s, a, b in keys]
    owner = np.array([k[0].split("/")[-1] for k in keys])
    base = cell_ba(y, ret_cap.argmax(1), cells)
    delta = np.zeros((len(orders), len(cells)))
    for oi, order in enumerate(orders):
        pos = np.searchsorted(banks["index"], ret_rows)
        members = {n: banks[n][order][pos] for n in names}
        pred = fuse(ret_cap, members, combo_list[choose])
        delta[oi] = 100 * (cell_ba(y, pred, cells) - base)
    return owner, base, delta


def ret_draws(owner, delta, rng):
    parts = []
    for person, share in (("P04", 0.2), ("P10", 0.8)):
        col = delta[:, owner == person]
        o = rng.integers(len(col), size=(DRAWS, len(col)))
        c = rng.integers(col.shape[1], size=(DRAWS, col.shape[1]))
        parts.append(share * col[o[:, :, None], c[:, None, :]].mean((1, 2)))
    return parts[0] + parts[1]


def run_cell(d, cap, ret_cap, ret_rows, banks_new, banks_ret, names, orders, seed):
    combo_list = combos(names)
    people, ncell, base, delta = new_deltas(d, cap, banks_new, names, combo_list, orders)
    nested = np.zeros((len(orders), len(people)))
    chosen = []
    for pi in range(len(people)):
        keep = [i for i in range(len(people)) if i != pi]
        ci = select(delta, combo_list, keep)
        chosen.append(combo_list[ci])
        nested[:, pi] = delta[ci, :, pi]
    rng = np.random.default_rng(seed)
    new = boot(nested, ncell, rng)
    mean_new = float(np.average(nested.mean(0), weights=ncell))
    eight = [i for i, p in enumerate(people) if p not in ("Dreyer2026Proteus/P04", "Dreyer2026Proteus/P10")]
    ci_ret = select(delta, combo_list, eight)
    owner, rbase, rdelta = returning(d, ret_rows, ret_cap, banks_ret, names, combo_list, orders, ci_ret)
    ret = ret_draws(owner, rdelta, rng)
    sealed = 0.57 * new + 0.43 * rng.permutation(ret)
    mean_ret = float(0.2 * rdelta[:, owner == "P04"].mean() + 0.8 * rdelta[:, owner == "P10"].mean())
    hind = int(np.argmax(delta.mean((1, 2))))
    return dict(orders=orders, members=list(names),
                held_new=dict(delta_pp=mean_new, p80=interval(new), by_person={p.split("/")[-1]: float(v)
                              for p, v in zip(people, nested.mean(0))}, lopo_weights=chosen),
                returning=dict(delta_pp=mean_ret, p80=interval(ret), weight=combo_list[ci_ret],
                               P04=float(rdelta[:, owner == "P04"].mean()), P10=float(rdelta[:, owner == "P10"].mean())),
                sealed_weighted=dict(delta_pp=0.57 * mean_new + 0.43 * mean_ret, p80=interval(sealed)),
                hindsight_best_weight=combo_list[hind], hindsight_new_delta_pp=float(np.average(
                    delta[hind].mean(0), weights=ncell)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("public", type=Path)
    p.add_argument("cap_full0", type=Path)
    p.add_argument("cap_ret", type=Path)
    p.add_argument("reve", type=Path, help="dir with ensemble_C_{held_new,returning}.npz")
    p.add_argument("cbramod", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    d = metadata(a.public)
    with np.load(a.cap_full0) as z:
        cap = posterior(z["stock|terminal"])
        delivery = z["delivery"]
    with np.load(a.cap_ret) as z:
        ret_rows, ret_cap = z["rows"], posterior(z["stock_scores"])
        require((ret_cap.argmax(1) == z["stock_labels"]).all(), "returning stock labels differ from scores")
    banks_new, banks_ret = {}, {}
    for name, path in (("REVE", a.reve), ("CBraMod", a.cbramod)):
        with np.load(path / "ensemble_C_held_new.npz") as z:
            np.testing.assert_array_equal(z["index"], np.arange(len(d["y"])))
            np.testing.assert_array_equal(z["full0_delivery"], delivery)
            banks_new[name] = {o: z[o] for o in ORDERS}
        with np.load(path / "ensemble_C_returning.npz") as z:
            require(set(z["index"]) == set(ret_rows.tolist()), "returning row sets differ")
            banks_ret[name] = {o: z[o] for o in ORDERS}
            banks_ret["index"] = z["index"]
    result = dict(screen_full0=screen(d, cap, {n: banks_new[n]["full0"] for n in banks_new}), cells={})
    for names in (("REVE",), ("CBraMod",), ("REVE", "CBraMod")):
        for tag, orders in (("full0_exact", ["full0"]), ("orders0-7_descriptive", ORDERS)):
            result["cells"]["+".join(names) + "|" + tag] = run_cell(
                d, cap, ret_cap, ret_rows, banks_new, banks_ret, names, orders, 20261119)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
