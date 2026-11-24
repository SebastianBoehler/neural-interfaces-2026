"""Official cell BA and paired people/order intervals from frozen outer-system decisions."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
from benchmark_utils.metrics import balanced_accuracy

ARMS = ("stock", "new", "heldnew", "strict", "ridge", "neural", "loose", "hybrid")
STATS = (*ARMS, "correct8", "correct20", "correct50", "correct", "wrong", "miss")
NBOOT = 20000


def interval(point, draws):
    return np.round([point, *np.quantile(draws, [.1, .9])], 4).tolist()


class Labels:
    def __init__(self, b10):
        self.tables = {}
        for group, name in (("pub", "data/proteus_windows.npz"), ("sch", "data/scherer2015/windows.npz")):
            with np.load(ROOT / name) as z:
                self.tables[group] = (z["y"], z["interface"])
        self.b10 = b10

    def get(self, person, tag, rows):
        if person.startswith("net_"):
            archive = self.b10 / ("v66_cp1_cap250_k" + person.split("_p")[1]) / "ret.npz"
            prefix = tag.replace("_", "|")
            with np.load(archive) as z:
                np.testing.assert_array_equal(rows, z[prefix + "|rows"])
                # Binary comparison: MI -> 0 and rest -> 1 on both sides.
                return 1 - z[prefix + "|y"], np.repeat("NETBCI", len(rows))
        y, iface = self.tables[person[:3]]
        return y[rows], iface[rows]


def collect(files, labels, arms=ARMS, gate="strict"):
    records = []
    for path in files:
        parts = path.stem.split("_")
        person = "_".join(parts[1:3])
        with np.load(path) as z:
            routing = json.loads(str(z["routing"]))
            for tag, routes in routing.items():
                def get(k):
                    return z[tag + "|" + k]
                rows = get("rows")
                y, iface = labels.get(person, tag, rows)
                np.testing.assert_array_equal(get("zero"), get("new"))
                scheme, seed, day = tag.split("_")
                preds = {a: get(a) for a in arms}
                # Independently check every closed prefix of the executed switches.
                for arm, route in ((gate, gate), ("loose", "loose"), ("hybrid", gate), ("protected", gate)):
                    if arm not in preds:
                        continue
                    at = routes[route][0]
                    end = len(rows) if at < 0 else at
                    base = "stock" if arm in ("hybrid", "protected") else "new"
                    np.testing.assert_array_equal(preds[arm][:end], preds[base][:end])
                for a in arms:
                    assert len(preds[a]) == len(rows) and set(preds[a]).issubset({0, 1, 2})
                changed = {a: int(np.sum(preds[a] != preds["stock"])) for a in arms}
                if person.startswith("net_"):
                    preds = {a: (v != 0).astype(int) for a, v in preds.items()}
                ba = {a: np.array([100 * balanced_accuracy(y[iface == i], v[iface == i])
                                   for i in np.unique(iface)]) for a, v in preds.items()}
                records.append(dict(person=person, day=int(day), order=(scheme, int(seed)), ba=ba, routes=routes,
                                    changes=changed, n=len(rows)))
    return records


def arrays(records, gate="strict"):
    arms = tuple(records[0]["ba"])
    persons = sorted({r["person"] for r in records})
    orders = sorted({r["order"] for r in records})
    fields = [*arms, *STATS[len(ARMS):]]
    total = np.zeros((len(persons), len(orders), len(fields)))
    counts = np.zeros_like(total)
    for r in records:
        i, j = persons.index(r["person"]), orders.index(r["order"])
        at, who = r["routes"][gate]
        correct = who == r["person"]
        vals = [correct and at < n for n in (8, 20, 50)] + [correct, bool(who) and not correct, not who]
        for k, a in enumerate(arms):
            total[i, j, k] += r["ba"][a].sum()
            counts[i, j, k] += len(r["ba"][a])
        total[i, j, len(arms):] += 100 * np.array(vals, dtype=float)
        counts[i, j, len(arms):] += 1
    assert (counts > 0).all()
    return persons, orders, total, counts


def sample(records, seed=20261120, gate="strict"):
    persons, orders, sums, counts = arrays(records, gate)
    point = sums.sum((0, 1)) / counts.sum((0, 1))
    slices = [np.flatnonzero([o[0] == g for o in orders]) for g in sorted({s for s, _ in orders})]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(NBOOT // 250):
        pi = rng.integers(len(persons), size=(250, len(persons), 1))
        oi = np.concatenate([rng.choice(idx, size=(250, len(persons), len(idx))) for idx in slices], 2)
        draws.append(sums[pi, oi].sum((1, 2)) / counts[pi, oi].sum((1, 2)))
    return point, np.concatenate(draws)


def summary(records, gate="strict"):
    arms = tuple(records[0]["ba"])
    stats = (*arms, *STATS[len(ARMS):])
    point, draws = sample(records, gate=gate)
    lookup = dict(zip(stats, point))
    boot = dict(zip(stats, draws.T))
    result = {k: interval(lookup[k], boot[k]) for k in stats}
    for baseline in ("stock", "new"):
        result["delta_" + baseline] = {k: interval(lookup[k] - lookup[baseline], boot[k] - boot[baseline])
                                       for k in arms}
    result["changes_vs_stock"] = {a: sum(r["changes"][a] for r in records) for a in arms}
    per = {}
    for p in sorted({r["person"] for r in records}):
        subset = [r for r in records if r["person"] == p]
        per[p] = {}
        for day in sorted({r["day"] for r in subset}):
            group = [r for r in subset if r["day"] == day]
            ages = [a + 1 for r in group for a, who in [r["routes"][gate]] if who == p]
            per[p][day] = dict(n=len(group), correct=len(ages),
                               wrong=sum(bool(r["routes"][gate][1]) and r["routes"][gate][1] != p
                                         for r in group), ages=ages)
    result["recognition_by_person_day"] = per
    result["n_stream_orders"] = len(records)
    result["n_windows"] = sum(r["n"] for r in records)
    ages = [a + 1 for r in records for a, who in [r["routes"][gate]] if who == r["person"]]
    result["correct_age_min_med_max"] = [min(ages), float(np.median(ages)), max(ages)] if ages else []
    return result


def main(args):
    labels = Labels(args.b10)
    groups = defaultdict(list)
    hashes = hashlib.sha256()
    for path in sorted(args.pred.glob("*.npz")):
        if path.name.startswith("pred_pop"):
            continue
        name = "_".join(path.stem.split("_")[:2])
        if name not in ("return_pub", "return_sch", "return_net", "new_pub", "new_sch", "new_net",
                        "publiconly_sch", "wrong_sch", "single_sch", "single_net"):
            raise ValueError(path)
        groups[name].append(path)
        hashes.update(path.name.encode() + path.read_bytes())
    counts = {"return_pub": 2, "return_sch": 9, "return_net": 19, "new_pub": 10, "new_sch": 9,
              "new_net": 19, "publiconly_sch": 9, "wrong_sch": 144, "single_sch": 9, "single_net": 19}
    assert {k: len(v) for k, v in groups.items()} == counts
    result, recs = {}, {}
    gate = "wave" if args.wave else "strict"
    arms = ("stock", "new", "heldnew", "wave", "protected", "oracle") if args.wave else ARMS
    from verify import coverage, public_reference
    for group, files in sorted(groups.items()):
        recs[group] = collect(files, labels, arms, gate)
        coverage(group, recs[group])
        result[group] = summary(recs[group], gate)
        if group in ("new_pub", "return_pub"):
            role = "new" if group == "new_pub" else "ret"
            result[group]["stock_native_ages"] = public_reference(args.stock, recs[group], role)
    for p in ("pub_P04", "pub_P10"):
        result[p] = summary([r for r in recs["return_pub"] if r["person"] == p], gate)
    from weighting import projections
    result["sealed_sensitivity"] = projections(recs, gate)
    result["data_sha256"] = hashes.hexdigest()
    args.dest.write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wave", action="store_true")
    p.add_argument("pred", type=Path)
    p.add_argument("b10", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--stock", type=Path, required=True)
    main(p.parse_args())
