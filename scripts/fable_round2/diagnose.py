"""P1 residual AUC, P2 oracle/T, P3 delivery deciles on verified fold tapes."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from folds import D, FOLDS, ORDERS, eye_mean, pinned_pair, standardise, streams

PERMS = list(itertools.permutations(range(3)))
SWAP = (0, 2, 1)


def auc(y, z, mask):
    result = roc_auc_score(y[mask] == 2, z[mask]) if len(np.unique(y[mask])) == 2 else None
    return {"auc": result, "n": int(mask.sum())}


def residual(y, z, lead):
    pair = y > 0
    return {
        "all": auc(y, z, pair),
        "correct": auc(y, z, pair & (lead == y)),
        "confused": auc(y, z, pair & (lead > 0) & (lead != y)),
    }


def statistic(g, pred, gap=None):
    if gap is not None and abs(gap) < 0.3:
        # Compute raw T for reporting, even when enrolled evidence is too weak to act.
        gap = None
    z = standardise(g)
    sub, word = z[pred == 1], z[pred == 2]
    if min(len(sub), len(word)) < 2:
        return None
    pooled = (np.sum((sub - sub.mean()) ** 2) + np.sum((word - word.mean()) ** 2)) / (len(sub) + len(word) - 2)
    if pooled <= 0:
        return None
    t = (word.mean() - sub.mean()) / np.sqrt(pooled * (1 / len(sub) + 1 / len(word)))
    return float(t if gap is None else t * np.sign(gap))


def arbiter(y, g, pred, gap):
    scores = [int(np.count_nonzero(np.asarray(perm)[pred] == y)) for perm in PERMS]
    best = int(np.argmax(scores))
    exact_tie = scores[best] > scores[0] and scores.count(scores[best]) > 1
    first, consecutive = None, 0
    if gap is None or abs(gap) >= 0.3:
        for t in range(40, len(y) + 1, 10):
            value = statistic(g[:t], pred[:t], gap)
            consecutive = consecutive + 1 if value is not None and value <= -3 else 0
            if consecutive == 2:
                first = t
                break
    return dict(oracle=PERMS[best], oracle_gain=100 * (scores[best] - scores[0]) / len(y),
                tied_nonidentities=exact_tie, oracle_swap=PERMS[best] == SWAP and not exact_tie,
                T={str(t): statistic(g[:t], pred[:t], gap) for t in (60, 120, len(y))},
                enrolled_gap=gap, first_trigger=first)


def decile_table(data, cases):
    all_rows = np.concatenate([c["rows"] for c in cases])
    y = data["y"][all_rows]
    pred = np.concatenate([c["pred"] for c in cases])
    decile = np.concatenate([10 * np.arange(len(c["rows"])) // len(c["rows"]) for c in cases])
    weight = []
    session = {}
    for case in cases:
        rows = case["rows"]
        y_stream, iface = data["y"][rows], data["interface"][rows]
        keys = np.char.add(iface.astype(str), y_stream.astype(str))
        w = np.asarray([1 / np.count_nonzero(keys == key) for key in keys])
        weight.extend(w)
        correct = case["pred"] == y_stream[:, None]
        session.setdefault(case["stream"], []).append(100 * np.average(correct, weights=w, axis=0))
    correct = pred == y[:, None]
    weight = np.asarray(weight)
    summary = {"accuracy": [], "cell_weighted": []}
    for j in range(10):
        mask = decile == j
        summary["accuracy"].append((100 * correct[mask].mean(0)).tolist())
        summary["cell_weighted"].append((100 * np.average(correct[mask], weights=weight[mask], axis=0)).tolist())
    summary["session"] = {s: np.mean(a, axis=0).tolist() for s, a in session.items()}
    return summary


def check_smoke(data, g):
    fold, stream, order = "new_P01", "P01|1", ORDERS[0]
    rows = D.plans(data, order, [stream])[stream]
    base, lead, _ = pinned_pair(fold, order, stream, rows)
    assert np.any(base != lead)
    assert not np.array_equal(rows, np.sort(rows))
    a = residual(data["y"][rows], standardise(g[rows]), lead)
    assert a["all"]["n"] > a["confused"]["n"] > 0
    toy_y = np.repeat([0, 1, 2], 60)
    shuffle = np.random.default_rng(0).permutation(len(toy_y))
    toy_y = toy_y[shuffle]
    toy_g = np.random.default_rng(1).normal(scale=0.1, size=len(toy_y)) + toy_y
    toy_pred = np.asarray(SWAP)[toy_y]
    assert arbiter(toy_y, toy_g, toy_pred, None)["first_trigger"] == 50
    assert arbiter(toy_y, toy_g, toy_pred, 0.1)["first_trigger"] is None
    assert arbiter(toy_y, toy_g, toy_pred, -0.6)["first_trigger"] is None
    assert arbiter(toy_y, toy_g, toy_y, None)["oracle"] == PERMS[0]
    print("PASS data, pinned order and arbiter controls", flush=True)


def diagnose(data, g, encoder):
    pooled = {}
    arbs = []
    deciles = {}
    one_session = {}
    for kind in ("new", "ret"):
        records = []
        for fold in (f for f in FOLDS if f.startswith(kind)):
            gap = None
            if kind == "ret":
                visit1 = data["stream"] == f"{fold.split('_')[1]}|1"
                y1, z1 = data["y"][visit1], standardise(g[visit1])
                gap = float(z1[y1 == 2].mean() - z1[y1 == 1].mean())
            for order in ORDERS:
                for stream, rows in D.plans(data, order, streams(data, fold)).items():
                    base, lead, weights_sha = pinned_pair(fold, order, stream, rows)
                    y, zg = data["y"][rows], g[rows]
                    record = dict(fold=fold, order=order, stream=stream, rows=rows, z=standardise(zg), lead=lead)
                    arbs.append(dict(fold=fold, order=order, stream=stream, **arbiter(y, zg, lead, gap)))
                    if encoder:
                        suffix = f"{order}_{stream.split('|')[1]}"
                        with np.load(encoder / f"{fold}.npz", allow_pickle=False) as z:
                            np.testing.assert_array_equal(z[f"rows_{suffix}"], rows)
                            assert str(z["weights_sha"]) == weights_sha
                            p = z[f"post_{suffix}"]
                            assert p.shape == (len(rows), 3) and np.isfinite(p).all()
                        record["pred"] = np.column_stack([base, p.argmax(1), lead])
                    records.append(record)
        grouped = {s: [r for r in records if r["stream"] == s] for s in sorted({r["stream"] for r in records})}
        def pool(group):
            return residual(data["y"][np.concatenate([r["rows"] for r in group])],
                            np.concatenate([r["z"] for r in group]), np.concatenate([r["lead"] for r in group]))
        pooled[kind] = pool(records)
        one_session[kind] = {s: pool(v) for s, v in grouped.items()}
        if encoder:
            deciles[kind] = decile_table(data, records)
    return dict(orders=ORDERS, residual=pooled, residual_sessions=one_session, arbiter=arbs, deciles=deciles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--out", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    data = D.load(args.data)
    g = eye_mean(data)
    if args.smoke:
        check_smoke(data, g)
    else:
        result = diagnose(data, g, args.encoder)
        args.out.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
