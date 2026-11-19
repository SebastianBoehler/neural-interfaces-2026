"""Authenticate complete causal worker predictions before scoring paired external cells."""
import hashlib
import json
from pathlib import Path

import numpy as np

from data import permutation
from protocol import score


DATA = Path("../neural-interfaces-2026/data/scherer2015/windows.npz")
PRIOR = Path("../ni26_cross411b_20261118")
ORDERS = (411, 1411)
ARMS = ("stock", "bounded")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bank(subject, day, groups):
    people = sorted(set(subject))
    source = sha(Path("source.sha256"))
    matrix = np.full((len(ARMS), len(ORDERS), len(subject)), -1, dtype=np.int8)
    deliveries = {order: permutation(groups, None, "full", order) for order in ORDERS}
    hashes = {}
    active = 0
    for i, person in enumerate(people):
        archive = PRIOR / f"cap_{i}.json"
        original = json.loads(archive.read_text())
        for a, arm in enumerate(ARMS):
            path = Path(f"{arm}_{i}.json")
            digest = sha(path)
            if digest != path.with_suffix(".sha256").read_text().strip():
                raise ValueError(f"invalid blind bank {path}")
            data = json.loads(path.read_text())
            if ((data["arm"], data["index"], data["source"], data["archive"]) != (arm, i, source, sha(archive))
                    or set(data["sessions"]) != set(original) or len(original) != 4):
                raise ValueError(f"worker provenance differs for {path}")
            hashes[path.name] = digest
            for key, reference in original.items():
                name, visit, order = key.split("|")
                j = ORDERS.index(int(order))
                rows = deliveries[int(order)][(subject == person) & (day == visit)]
                got = data["sessions"][key]
                if (name != person or rows.tolist() != got["rows"] or got["rows"] != reference["rows"]
                        or got["activity"] != reference["activity"] or got["activity"]["errors"]
                        or len(got["labels"]) != len(rows) or not np.isin(got["labels"], (0, 1, 2)).all()
                        or got["labels"][0] != reference["labels"][0]
                        or (matrix[a, j, rows] >= 0).any()):
                    raise ValueError(f"causal rows, reset, labels or work differ for {key}/{arm}")
                if arm == "stock" and got["labels"] != reference["labels"]:
                    raise ValueError(f"stock archive labels differ for {key}")
                if arm == "bounded":
                    if (set(got["adapters"]) != {str(k) for k in range(47, 53)} or
                            any(v["windows"] != len(rows) for v in got["adapters"].values())):
                        raise ValueError(f"incomplete bounded trace use for {key}")
                    active += sum(v["clipped"] for v in got["adapters"].values())
                matrix[a, j, rows] = got["labels"]
    if (matrix < 0).any() or not active or np.array_equal(matrix[0], matrix[1]):
        raise ValueError("incomplete or inactive bounded history")
    with Path("before_labels.json").open("x") as handle:
        json.dump(hashes, handle, sort_keys=True, indent=2)
    return people, matrix, active


def collect():
    with np.load(DATA, allow_pickle=False) as data:
        subject, visit, interface = (data[k] for k in ("subject", "session", "interface"))
    stream = np.char.add(np.char.add(subject, "|"), visit)
    people, matrix, active = load_bank(subject, visit, stream)
    with np.load(DATA, allow_pickle=False) as data:
        label = data["y"]
    meta = dict(stream=stream, interface=interface)
    cells = np.empty((len(ARMS), len(people), 2, len(ORDERS)))
    for a, decisions in enumerate(matrix):
        for j, pred in enumerate(decisions):
            for i, person in enumerate(people):
                for v, day in enumerate(sorted(set(visit[subject == person]))):
                    mask = (subject == person) & (visit == day)
                    part = score(label[mask], pred[mask], meta, mask)
                    if part["n_cells"] != 1 or len(np.unique(label[mask])) != 3:
                        raise ValueError("unexpected external cell")
                    cells[a, i, v, j] = 100 * part["balanced_accuracy_cell_mean"]
            complete = score(label, pred, meta, np.ones(len(label), bool))
            if complete["n_cells"] != 18 or abs(100 * complete["balanced_accuracy_cell_mean"]
                                                  - cells[a, :, :, j].mean()) > 1e-10:
                raise ValueError("official and paired external reductions differ")
    historical = json.loads((PRIOR / "result.json").read_text())["acc"][1]
    if abs(cells[0].mean() - historical) > 1e-10:
        raise ValueError("fresh exact stock differs from §411b")
    delta = cells[1] - cells[0]
    rng = np.random.default_rng(468)
    p = rng.integers(9, size=(10000, 9, 1, 1))
    d = rng.integers(2, size=(10000, 9, 2, 1))
    o = rng.integers(2, size=(10000, 9, 2, 2))
    interval = lambda values: np.quantile(values, (0.1, 0.9)).tolist()
    pooled = interval(delta[p, d, o].mean((1, 2, 3)))
    by_order = {order: dict(pp=float(delta[:, :, j].mean()), paired80=interval(
        delta[:, :, j][p[..., 0], d[..., 0]].mean((1, 2)))) for j, order in enumerate(ORDERS)}
    gain = pooled[0] > 0 and all(row["paired80"][0] > 0 for row in by_order.values())
    stock, bounded = matrix
    result = dict(scope="exact FULL-SYSTEM EXTERNAL", absolute=dict(zip(ARMS, cells.mean((1, 2, 3)).tolist())),
                  pp=float(delta.mean()), paired80=pooled, by_order=by_order,
                  people=dict(zip(people, delta.mean((1, 2)).tolist())),
                  outcome="advance" if gain else "refuted" if pooled[1] <= 0 else "undecided",
                  changed=int(np.count_nonzero(stock != bounded)),
                  corrections=int(np.count_nonzero((stock != label) & (bounded == label))),
                  regressions=int(np.count_nonzero((stock == label) & (bounded != label))),
                  clipped=active, source=sha(Path("source.sha256")), scorer=sha(Path("score.sha256")),
                  decision_lock=sha(Path("before_labels.json")), archive=sha(PRIOR / "result.json"))
    with Path("result.json").open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
    print(f"PASS F-C10 exact matched external score {sha(Path('result.json'))}", flush=True)


if __name__ == "__main__":
    collect()
