"""Seal the matched Scherer decisions before opening external labels."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

import protocol
from external_inputs import (ACTOR_SHA, ARMS, DATA, DATA_SHA, HERE, KINDS, METRIC_SHA, MODES, ORDERS, PEOPLE,
                             ROW_MAP, TOL, VISITS, ancestry, cohort, deliveries, delivery_receipt)
from held_external import paired
from replay_f13 import sha


def seal_banks(subject, stream):
    source = {}
    for seed in (0, 1):
        for kind, (path, meta) in paired(seed, smoke=False).items():
            key = f"s{seed}_{kind}"
            source[key] = dict(checkpoint=meta["checkpoint"], state=meta["state"], receipt=sha(path.with_suffix(".json")))
    scorer = sha(HERE / "score.sha256")
    banks, active, prefix_error, first, resets = {}, 0, 0.0, 0, 0
    decisions = np.full((len(ARMS), len(ORDERS), len(subject)), -1, dtype=np.int8)
    for i, person in enumerate(PEOPLE):
        path = HERE / "banks" / f"person_{i}.npz"
        digest = sha(path)
        if digest != path.with_suffix(".sha256").read_text().strip():
            raise ValueError(f"blind bank digest differs for {person}")
        rows = np.flatnonzero(subject == person)
        expected = dict(person=person, actor=ACTOR_SHA, scorer=scorer, dataset=DATA_SHA, held_labels_opened=False,
                        delivery=delivery_receipt(deliveries(stream, person)))
        with np.load(path, allow_pickle=False) as z:
            receipt = json.loads(str(z["metadata"]))
            if (any(receipt[k] != v for k, v in expected.items()) or not np.array_equal(z["index"], rows) or
                    set(receipt["evidence"]) != set(source) or set(z.files) != set(ARMS) | {"index", "metadata"}):
                raise ValueError(f"held bank provenance differs for {person}")
            result = np.stack([z[arm] for arm in ARMS])
        if result.shape != (len(ARMS), len(ORDERS), len(rows)) or not np.isin(result, (0, 1, 2)).all():
            raise ValueError(f"held bank has invalid decisions for {person}")
        person_active = 0
        for k, proof in receipt["evidence"].items():
            if (any(proof[a] != b for a, b in source[k].items()) or
                    (proof["first_windows_exact"], proof["reset_checks"]) != (4, 8) or
                    not 0 <= proof["prefix_max_abs"] <= TOL or proof["active_eval_clips"] < 0):
                raise ValueError(f"held source or isolation proof differs for {person}: {k}")
            prefix_error = max(prefix_error, proof["prefix_max_abs"])
            person_active += proof["active_eval_clips"]
            first += proof["first_windows_exact"]
            resets += proof["reset_checks"]
        if not person_active:
            raise ValueError(f"bounded history was inactive for {person}")
        active += person_active
        decisions[:, :, rows] = result
        banks[path.name] = digest
    if (decisions < 0).any():
        raise ValueError("missing external canonical rows")
    lock = HERE / "before_labels.json"
    with lock.open("x") as handle:
        json.dump(dict(actor=ACTOR_SHA, scorer=scorer, dataset=DATA_SHA, banks=banks, source=source),
                  handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return decisions, sha(lock), dict(clips=active, prefix_max_abs=prefix_error, first_windows_exact=first, resets=resets)


def labels_after_seal(locked_sha, subject, visit):
    if sha(HERE / "before_labels.json") != locked_sha:
        raise ValueError("external decisions must be sealed before labels")
    with np.load(DATA, allow_pickle=False) as z:
        labels, onset = z["y"], z["onset"]
    mapped = np.full(len(labels), -1)
    class_map = {4: 0, 2: 1, 1: 2}
    with gzip.open(ROW_MAP, "rt") as handle:
        for line in csv.DictReader(handle, delimiter="\t"):
            i = int(line["row"])
            value = class_map[int(line["raw_class"])]
            if (not 0 <= i < len(labels) or mapped[i] >= 0 or value != int(line["export_class"]) or
                    subject[i] != f"Scherer2015/{line['person']}" or visit[i] != line["session"] or
                    onset[i] != float(line["cue_s"])):
                raise ValueError("native Scherer row or class mapping differs")
            mapped[i] = value
    if not np.array_equal(mapped, labels):
        raise ValueError("exported Scherer classes disagree with all pinned raw class codes")
    return labels


def metric_cells(decisions, label, subject, visit, meta):
    if sha(Path(protocol.__file__)) != METRIC_SHA:
        raise ValueError("loaded official cell reduction differs")
    cells = np.empty((len(ARMS), len(PEOPLE), len(VISITS), len(ORDERS)))
    all_rows = np.ones(len(label), dtype=bool)
    for a, arm in enumerate(decisions):
        for o, pred in enumerate(arm):
            for p, person in enumerate(PEOPLE):
                for v, day in enumerate(VISITS):
                    mask = (subject == person) & (visit == day)
                    part = protocol.score(label[mask], pred[mask], meta, mask)
                    if part["n_cells"] != 1 or set(label[mask]) != {0, 1, 2}:
                        raise ValueError("external cell misses a class")
                    cells[a, p, v, o] = 100 * part["balanced_accuracy_cell_mean"]
            full = protocol.score(label, pred, meta, all_rows)
            if full["n_cells"] != 18 or abs(100 * full["balanced_accuracy_cell_mean"] - cells[a, :, :, o].mean()) > 1e-10:
                raise ValueError("official metric differs from the paired cells")
    return dict(zip(ARMS, cells, strict=True))


def comparisons(cells):
    rng = np.random.default_rng(15015)
    person = rng.integers(9, size=(10000, 9, 1, 1))
    visit = rng.integers(2, size=(10000, 9, 2, 1))
    order = rng.integers(2, size=(10000, 9, 2, 2))

    def describe(delta):
        draws = delta[person, visit, order].mean((1, 2, 3))
        by_order = {str(key): float(delta[:, :, i].mean()) for i, key in enumerate(ORDERS)}
        return dict(pp=float(delta.mean()), paired80=np.quantile(draws, (0.10, 0.90)).tolist(), by_order=by_order)

    contrasts = {}
    for seed in (0, 1):
        for mode in MODES:
            for control in ("scaled", "original"):
                prefix = f"s{seed}"
                contrasts[f"{prefix}_{mode}_dropout_minus_{control}"] = describe(
                    cells[f"{prefix}_dropout_{mode}"] - cells[f"{prefix}_{control}_{mode}"])
        for kind in KINDS:
            name = f"s{seed}_{kind}"
            contrasts[f"{name}_bounded_minus_ordinary"] = describe(cells[f"{name}_bounded"] - cells[f"{name}_ordinary"])

    def rule(control):
        intervals = [contrasts[f"s{seed}_bounded_dropout_minus_{control}"]["paired80"] for seed in (0, 1)]
        if all(a > 0 for a, _ in intervals):
            return "supported"
        if all(b <= 0 for _, b in intervals):
            return "refuted"
        return "undecided"

    return contrasts, {key: rule(key) for key in ("scaled", "original")}


def run():
    ancestry()
    _, subject, visit, interface, stream = cohort(waveforms=False)
    decisions, lock, guards = seal_banks(subject, stream)
    label = labels_after_seal(lock, subject, visit)
    cells = metric_cells(decisions, label, subject, visit, dict(stream=stream, interface=interface))
    contrasts, decisions = comparisons(cells)
    result = dict(scope="matched WHOLE EXTERNAL DEV on reused Scherer cohort", people=PEOPLE, visits=VISITS,
                  rows=len(label), orders=ORDERS, class_map={"4": "motor", "2": "subtraction", "1": "word"},
                  actor=ACTOR_SHA, scorer=sha(HERE / "score.sha256"), before_labels=lock, guards=guards,
                  accuracy={key: float(value.mean()) for key, value in cells.items()}, contrasts=contrasts,
                  verdict=decisions, cells={key: value.tolist() for key, value in cells.items()})
    path = HERE / "result.json"
    with path.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"PASS F15e matched WHOLE EXTERNAL DEV {sha(path)}", flush=True)


if __name__ == "__main__":
    run()
