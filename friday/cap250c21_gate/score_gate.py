"""Score only sealed complete output selection on returning and held-new tapes."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from eye_scope import DATA, DATA_SHA, pin, sha
from gate_inputs import metadata
from score_eye import DRAWS, METRIC, METRIC_SHA, SEED, point_interval, read_cells
from select_fbl2 import NAMES, ORDERS
from source_labels import read_training_labels


PAIRS = dict(selected_minus_stock=(0, 2), fbl2_minus_stock=(0, 1), selected_minus_fbl2=(1, 2))


def authenticated(name, scope):
    path = Path(f"{name}_selected.npz")
    pin(path, scope["bank"])
    if path.stat().st_mode & 0o222:
        raise ValueError("decision bank is mutable")
    with np.load(path, allow_pickle=False) as z:
        rows, use = z["rows"], z["eligible"]
        np.testing.assert_array_equal(use, (z["age"] >= 20) & ~z["recognized"])
        np.testing.assert_array_equal(z["selected"][use], z["fbl2"][use])
        np.testing.assert_array_equal(z["selected"][~use], z["stock"][~use])
        if len(rows) != scope["rows"] or use.sum() != scope["eligible"] or len(np.unique(rows)) != len(rows):
            raise ValueError("declared complete scope differs from sealed data")
        pred = np.column_stack([z[k].argmax(1) for k in NAMES])
    return rows, pred, use


def cases(rows, person, visit):
    changes = (person[1:] != person[:-1]) | (visit[1:] != visit[:-1])
    return np.split(rows, np.flatnonzero(changes) + 1)


def held_new(pred, y, person, visit, interface, metric):
    groups = np.stack((person, visit, interface), axis=1)
    sums, n, per = [], [], {}
    for who in sorted(set(person)):
        m = person == who
        cells = np.column_stack([metric.group_scores(metric.balanced_accuracy, y[m], p[m], groups[m]) for p in pred.T])
        sums.append(cells.sum(0))
        n.append(len(cells))
        per[who.split("/")[-1]] = dict(zip(NAMES, (100 * cells.mean(0)).tolist(), strict=True))
    total, weights = np.asarray(sums), np.asarray(n)
    if (len(weights), sum(weights)) != (10, 28):
        raise ValueError("official held-new grouping changed")
    draw = np.random.default_rng(SEED).integers(0, 10, (DRAWS, 10))
    point = 100 * total.sum(0) / weights.sum()
    boot = 100 * total[draw].sum(1) / weights[draw].sum(1)[:, None]
    comparisons = {name: dict(pp=float(point[r] - point[l]), paired80=np.percentile(boot[:, r] - boot[:, l], (10,90)).tolist())
                   for name, (l, r) in PAIRS.items()}
    return dict(accuracy=dict(zip(NAMES, point.tolist(), strict=True)), comparisons=comparisons, people=per)


def returning(rows, pred, y, person, visit, interface, metric):
    seq = cases(rows, person, visit)
    cells, sessions, _ = read_cells(rows, pred, seq, y, interface, metric.balanced_accuracy)
    weights = dict(P04=(1., 0.), P10=(0., 1.), official_equal=(.5, .5), weighted_20_80=(.2,.8))
    profiles = {name: {k: point_interval(cells[:, :, pair], w, np.random.default_rng(SEED)) for k, pair in PAIRS.items()}
                for name, w in weights.items()}
    return dict(comparisons=profiles, sessions=sessions)


def analyze(name, rows, pred, use, labels, metric):
    person, visit, interface = metadata(rows)
    y = labels[rows]
    if name == "returning":
        result = returning(rows, pred, y, person, visit, interface, metric)
    else:
        result = held_new(pred, y, person, visit, interface, metric)
    activity = {}
    for tag, mask in (("eligible", use), ("protected", ~use)):
        q, g = pred[mask], y[mask]
        stock, fbl2, selected = (v == g for v in q.T)
        activity[tag] = {key: dict(rescue=int((~stock & ok).sum()), loss=int((stock & ~ok).sum()),
                                   changed=int((q[:, 0] != q[:, i]).sum()))
                         for i, key, ok in ((1, "fbl2", fbl2), (2, "selected", selected))}
    result["activity"] = activity
    return result


def main():
    receipt = Path("complete_select.json")
    proof = json.loads(receipt.read_text())
    source = sha("source.sha256")
    if (proof["source"] != source or proof["held_labels_opened"] or proof["accuracy_computed"]
            or proof["arms"] != list(NAMES) or set(proof["scopes"]) != set(ORDERS) or receipt.stat().st_mode & 0o222):
        raise ValueError("complete F-C21g blind seal required")
    banks = {name: authenticated(name, proof["scopes"][name]) for name in ORDERS}
    pin(METRIC, METRIC_SHA)
    spec = importlib.util.spec_from_file_location("ni26_gate_metric", METRIC)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    pin(DATA, DATA_SHA)
    labels = read_training_labels(DATA, np.arange(2655), 2655)
    scores = {name: analyze(name, *data, labels, metric) for name, data in banks.items()}
    advancement = {}
    for name in ("within0", "full0"):
        scoreset = scores[name]["comparisons"]
        post, always = (scoreset[k] for k in ("selected_minus_stock", "fbl2_minus_stock"))
        ratio = post["pp"] / always["pp"]
        advancement[name] = dict(retained=ratio, qualifies=bool(ratio >= .6 and post["paired80"][0] > 0))
    output = Path("complete_select_score.json")
    with output.open("x") as f:
        json.dump(dict(scope="FULL-SYSTEM DEV banked terminal choice, separate conditional returning and held-new",
                       source=source, blind=sha(receipt), scores=scores, advancement=advancement),
                  f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    output.chmod(0o444)
    print("PASS separate complete F-C21g score", sha(output), advancement, flush=True)


if __name__ == "__main__":
    main()
