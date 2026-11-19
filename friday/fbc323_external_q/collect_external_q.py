"""Authenticate the same-weight Scherer decision lock, then reduce official paired cells."""
import json
from pathlib import Path

import numpy as np

import protocol
from collect_external import labels_after_seal
from external_inputs import DATA_SHA, METRIC_SHA, ORDERS, PEOPLE, TOL, VISITS, ancestry, cohort, deliveries, delivery_receipt
from held_external_q import ARMS, FEATURES, F15E, F15E_BANK, HERE, KINDS, MODES, old_lock, sources
from replay_f13 import sha


F16E = HERE.parent / "ni26_friday_s1c_distill_external_20261118"
F16E_RESULT = "75cbbaf0d2df958cb2a353c036629e6dc0bc9253465d5e30156833d428234a5f"


def seal(subject, stream):
    history = old_lock()
    source = {name: proof for name, (_, _, proof) in sources((0, 1), history).items()}
    scorer = sha(HERE / "score.sha256")
    decisions = np.full((len(ARMS), len(ORDERS), len(subject)), -1, dtype=np.int8)
    fingerprints = {}
    first = resets = matches = activity = 0
    prefix_error, effect = 0., 0.
    for i, person in enumerate(PEOPLE):
        path = HERE / "banks" / f"person_{i}.npz"
        digest = sha(path)
        if digest != path.with_suffix(".sha256").read_text().strip():
            raise ValueError("unsealed paired predictions")
        fingerprints[path.name] = digest
        rows = np.flatnonzero(subject == person)
        parent = F15E / "banks" / path.name
        archive = sha(parent)
        if archive != history["banks"][path.name]:
            raise ValueError("original F15e person bank differs")
        fixed = dict(person=person, data=DATA_SHA, scorer=scorer, archive=archive, held_labels_opened=False,
                     delivery=delivery_receipt(deliveries(stream, person)))
        number = len(MODES) * len(ORDERS) * len(rows)
        with np.load(path, allow_pickle=False) as z, np.load(parent, allow_pickle=False) as previous:
            meta = json.loads(str(z["metadata"]))
            if (any(meta[k] != v for k, v in fixed.items()) or not np.array_equal(z["index"], rows) or
                    not np.array_equal(previous["index"], rows) or set(z.files) != set(ARMS) | {"index", "metadata"} or
                    set(meta["evidence"]) != set(source)):
                raise ValueError("wrong same-weight person provenance or canonical rows")
            for checkpoint, proof in meta["evidence"].items():
                if (any(proof[k] != v for k, v in source[checkpoint].items()) or (proof["first"], proof["resets"]) != (8, 16)
                        or proof["full_q_archive_matches"] != number or not 0 <= proof["prefix_max"] <= TOL
                        or not np.isfinite(proof["q_effect"]) or proof["q_effect"] <= 1e-4 or proof["clipped"] < 0):
                    raise ValueError("wrong checkpoint, active q, first window or reset evidence")
                first += proof["first"]
                resets += proof["resets"]
                matches += number
                activity += proof["clipped"]
                prefix_error = max(prefix_error, proof["prefix_max"])
                effect = max(effect, proof["q_effect"])
                for mode in MODES:
                    arm = f"{checkpoint}_{mode}"
                    if not np.array_equal(z[f"{arm}_on"], previous[arm]):
                        raise ValueError("full-q archive decision recovery differs")
            values = np.stack([z[k] for k in ARMS])
        if values.shape != (len(ARMS), len(ORDERS), len(rows)) or not np.isin(values, (0, 1, 2)).all():
            raise ValueError("noncanonical blind outputs")
        decisions[:, :, rows] = values
    if (decisions < 0).any() or not activity:
        raise ValueError("missing blind same-weight decisions or bounded work")
    path = HERE / "before_labels.json"
    with path.open("x") as handle:
        json.dump(dict(banks=fingerprints, source=source, prior=F15E_BANK, scorer=scorer, data=DATA_SHA), handle,
                  sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return decisions, sha(path), dict(first=first, resets=resets, full_q_matches=matches, max_q_effect=effect,
                                      active_clips=activity, prefix_max_abs=prefix_error)


def official_cells(decisions, y, subject, visit, meta):
    if sha(Path(protocol.__file__)) != METRIC_SHA:
        raise ValueError("official cell scorer differs")
    cell = np.empty((len(ARMS), len(PEOPLE), len(VISITS), len(ORDERS)))
    masks = [(person, day, (subject == name) & (visit == v)) for person, name in enumerate(PEOPLE)
             for day, v in enumerate(VISITS)]
    all_rows = np.ones(len(y), bool)
    for a, predictions in enumerate(decisions):
        for o, pred in enumerate(predictions):
            for p, v, mask in masks:
                score = protocol.score(y[mask], pred[mask], meta, mask)
                if score["n_cells"] != 1 or set(y[mask]) != {0, 1, 2}:
                    raise ValueError("missing an external balanced-accuracy cell")
                cell[a, p, v, o] = 100 * score["balanced_accuracy_cell_mean"]
            reference = protocol.score(y, pred, meta, all_rows)
            if reference["n_cells"] != len(masks) or abs(100 * reference["balanced_accuracy_cell_mean"]
                                                        - cell[a, :, :, o].mean()) > 1e-10:
                raise ValueError("official reduction disagrees with the external cells")
    return dict(zip(ARMS, cell, strict=True))


def comparisons(cells):
    rng = np.random.default_rng(15015)
    p = rng.integers(len(PEOPLE), size=(10000, len(PEOPLE), 1, 1))
    v = rng.integers(len(VISITS), size=(10000, len(PEOPLE), len(VISITS), 1))
    o = rng.integers(len(ORDERS), size=(10000, len(PEOPLE), len(VISITS), len(ORDERS)))
    contrasts, verdict = {}, {}
    for kind in KINDS:
        for seed in (0, 1):
            for mode in MODES:
                arm = f"s{seed}_{kind}_{mode}"
                delta = cells[f"{arm}_zero"] - cells[f"{arm}_on"]
                paired = np.quantile(delta[p, v, o].mean((1, 2, 3)), (0.1, 0.9)).tolist()
                contrasts[arm] = dict(pp=float(delta.mean()), paired80=paired,
                    orders={str(n): float(delta[:, :, j].mean()) for j, n in enumerate(ORDERS)})
        limits = [contrasts[f"s{s}_{kind}_bounded"]["paired80"] for s in (0, 1)]
        verdict[kind] = ("supported" if all(a > 0 for a, _ in limits) else
                         "refuted" if all(b <= 0 for _, b in limits) else "undecided")
    return contrasts, verdict


def run():
    ancestry()
    _, subject, visit, interface, stream = cohort(waveforms=False)
    output, locked, guards = seal(subject, stream)
    if sha(HERE / "before_labels.json") != locked:
        raise ValueError("new same-weight lock changed before label access")
    label = labels_after_seal(F15E_BANK, subject, visit)
    cells = official_cells(output, label, subject, visit, dict(stream=stream, interface=interface))
    contrast, verdict = comparisons(cells)
    path = F16E / "result.json"
    if sha(path) != F16E_RESULT:
        raise ValueError("separate contextual hard-CE result differs")
    context = json.loads(path.read_text())["accuracy"]
    result = dict(scope="same-weight WHOLE EXTERNAL DEV on reused Scherer", people=PEOPLE, visits=VISITS,
                  orders=ORDERS, labels=len(label), before_labels=locked, guards=guards, contrast=contrast, verdict=verdict,
                  f16e_hard_context={f"s{s}_{m}": context[f"s{s}_hard_{m}"] for s in (0, 1) for m in MODES},
                  absolute={k: float(x.mean()) for k, x in cells.items()}, cells={k: x.tolist() for k, x in cells.items()})
    score = HERE / "result.json"
    with score.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"PASS F15eq same-weight WHOLE EXTERNAL DEV {sha(score)}", flush=True)


if __name__ == "__main__":
    run()
