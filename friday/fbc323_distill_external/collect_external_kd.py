"""Seal all external raw-only decisions before checking labels and paired gains."""
import csv
import gzip
import json

import numpy as np

from anchors import MODES, sha
from collect_external import metric_cells
from external_inputs import ARMS as F15E_ARMS
from external_inputs import (DATA, DATA_SHA, ORDERS, PEOPLE, ROW_MAP, SOURCE, TOL, VISITS, ancestry, cohort, deliveries,
                             delivery_receipt)
from external_source import HERE
from held_external_kd import ARMS
from receipts_external import ACTOR, F15E, paired


F15E_RESULT = "604eb04ca1cd5536c96ef73d84c138d799d4fd5defd1a4cacf75559d3606312e"


def seal_banks(subject, stream):
    source = {}
    for seed in (0, 1):
        for kind, (path, meta) in paired(seed, smoke=False).items():
            source[f"s{seed}_{kind}"] = dict(checkpoint=meta["checkpoint"], state=meta["state"],
                                            receipt=sha(path.with_suffix(".json")),
                                            teacher_bank=meta["teacher_bank"], donor_plan=meta["donor_plan"])
    scorer = sha(HERE / "score.sha256")
    predictions = np.full((len(ARMS), len(ORDERS), len(subject)), -1, dtype=np.int8)
    fingerprints = {}
    first, resets, clips, error = 0, 0, 0, 0.
    for i, person in enumerate(PEOPLE):
        path = HERE / "banks" / f"person_{i}.npz"
        digest = sha(path)
        if digest != path.with_suffix(".sha256").read_text().strip():
            raise ValueError("external blind decisions changed")
        fingerprints[path.name] = digest
        test = np.flatnonzero(subject == person)
        fixed = dict(person=person, actor=ACTOR, scorer=scorer, data=DATA_SHA, held_labels_opened=False,
                     delivery=delivery_receipt(deliveries(stream, person)))
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if (any(meta[k] != v for k, v in fixed.items()) or not np.array_equal(z["index"], test) or
                    set(z.files) != set(ARMS) | {"metadata", "index"} or set(meta["evidence"]) != set(source)):
                raise ValueError("external held provenance or row alignment differs")
            bank = np.stack([z[arm] for arm in ARMS])
        if bank.shape != (len(ARMS), len(ORDERS), len(test)) or not np.isin(bank, (0, 1, 2)).all():
            raise ValueError("invalid external held decisions")
        for key, p in meta["evidence"].items():
            if (any(p[k] != v for k, v in source[key].items()) or (p["first_windows_exact"], p["reset_checks"]) != (4, 8)
                    or p["clips"] < 0 or not 0 <= p["prefix_max_abs"] <= TOL):
                raise ValueError("source or reset proof differs for the external bank")
            first += 4
            resets += 8
            clips += p["clips"]
            error = max(error, p["prefix_max_abs"])
        predictions[:, :, test] = bank
    if (predictions < 0).any() or clips <= 0:
        raise ValueError("incomplete or inactive external predictions")
    lock = HERE / "before_labels.json"
    with lock.open("x") as handle:
        json.dump(dict(banks=fingerprints, source=source, scorer=scorer, actor=ACTOR, data=DATA_SHA),
                  handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return predictions, sha(lock), dict(first=first, resets=resets, active_clips=clips, prefix_max_abs=error)


def external_labels(lock, subject, visit):
    if sha(HERE / "before_labels.json") != lock:
        raise ValueError("blind decision lock differs before external labels")
    with np.load(DATA, allow_pickle=False) as z:
        y, onset = z["y"], z["onset"]
    axes = {4: 0, 2: 1, 1: 2}
    mapped = np.full(len(y), -1)
    with gzip.open(ROW_MAP, "rt") as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            row = int(r["row"])
            target = axes[int(r["raw_class"])]
            if (not 0 <= row < len(y) or mapped[row] != -1 or target != int(r["export_class"]) or
                    subject[row] != f"Scherer2015/{r['person']}" or visit[row] != r["session"] or
                    onset[row] != float(r["cue_s"])):
                raise ValueError("original external row, cue or task code differs")
            mapped[row] = target
    if not np.array_equal(mapped, y):
        raise ValueError("exported external labels differ from verified raw source codes")
    return y


def comparisons(cells):
    rng = np.random.default_rng(15015)
    p = rng.integers(len(PEOPLE), size=(10000, len(PEOPLE), 1, 1))
    v = rng.integers(len(VISITS), size=(10000, len(PEOPLE), len(VISITS), 1))
    o = rng.integers(len(ORDERS), size=(10000, len(PEOPLE), len(VISITS), len(ORDERS)))

    def contrast(delta):
        draws = delta[p, v, o].mean((1, 2, 3))
        return dict(pp=float(delta.mean()), paired80=np.quantile(draws, (0.1, 0.9)).tolist(),
                    orders={str(k): float(delta[:, :, j].mean()) for j, k in enumerate(ORDERS)})

    result = {}
    for seed in (0, 1):
        for mode in MODES:
            for a, b in (("trial", "hard"), ("trial", "deranged"), ("deranged", "hard")):
                result[f"s{seed}_{mode}_{a}_minus_{b}"] = contrast(cells[f"s{seed}_{a}_{mode}"] - cells[f"s{seed}_{b}_{mode}"])
    outcome = {}
    for key in ("hard", "deranged"):
        limits = [result[f"s{i}_bounded_trial_minus_{key}"]["paired80"] for i in (0, 1)]
        outcome[key] = ("supported" if all(a > 0 for a, _ in limits) else
                        "refuted" if all(b <= 0 for _, b in limits) else "undecided")
    return result, outcome


def collect():
    ancestry()
    _, subject, visit, interface, stream = cohort(waveforms=False)
    pred, seal, guards = seal_banks(subject, stream)
    with np.load(SOURCE, allow_pickle=False) as z:
        source_labels = z["y"]
    for seed in (0, 1):
        paired(seed, False, source_labels=source_labels)
    y = external_labels(seal, subject, visit)
    legacy = metric_cells(pred, y, subject, visit, dict(stream=stream, interface=interface))
    cells = {current: legacy[prior] for current, prior in zip(ARMS, F15E_ARMS, strict=True)}
    paired_results, verdict = comparisons(cells)
    path = F15E / "result.json"
    if sha(path) != F15E_RESULT:
        raise ValueError("contextual full-feature F15e teacher score differs")
    available = json.loads(path.read_text())["accuracy"]
    teacher = {f"s{seed}_{mode}": available[f"s{seed}_dropout_{mode}"] for seed in (0, 1) for mode in MODES}
    result = dict(scope="matched raw-only WHOLE EXTERNAL DEV on reused Scherer", people=PEOPLE, visits=VISITS,
                  orders=ORDERS, rows=len(y), guards=guards, verdict=verdict, contrasts=paired_results,
                  accuracy={k: float(v.mean()) for k, v in cells.items()}, cells={k: v.tolist() for k, v in cells.items()},
                  full_q_teacher=teacher, before_labels=seal, actor=ACTOR, scorer=sha(HERE / "score.sha256"))
    output = HERE / "result.json"
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"PASS F16e matched raw-only WHOLE EXTERNAL DEV {sha(output)}", flush=True)


if __name__ == "__main__":
    collect()
