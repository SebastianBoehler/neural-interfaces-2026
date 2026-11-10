"""Matched feedback crops with fixed source fits and two held-out slices."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from head import features, fit, logits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "brainhero_seed_wave"))
from run import (EVAL, LATE, SEEDS, SINGLES, compare, load_trace, manifest, metadata,
                 partition, protocol, read_eeg, read_labels, save, source_rows, verify_receipts)

ARMS = ("pre", "post", "full")
INTERFACES = ("BrainHero", "Graz")
MODES = ("real", "sham")
WEIGHT = 0.15
TASK = {"AcquisitionBH": ("BrainHero", "sham"), "AcquisitionGraz": ("Graz", "sham"),
        "OnlineRawBH": ("BrainHero", "real"), "OnlineRawGraz": ("Graz", "real")}


def modes(meta, rows):
    with np.load(protocol.DATA, allow_pickle=False) as data:
        tasks = data["task"][rows]
    context, mode = np.array([TASK[t] for t in tasks]).T
    np.testing.assert_array_equal(meta["interface"][rows], context)
    return mode


def bootstrap(meta, rows, y, mode, fit_rows, seed, fold):
    rng = np.random.default_rng(np.random.SeedSequence((seed, fold)))
    cells = np.column_stack([meta["subject"][rows], meta["interface"][rows], mode, y])
    _, group = np.unique(cells[fit_rows], axis=0, return_inverse=True)
    draws = [rng.choice(fit_rows[group == i], size=int((group == i).sum()), replace=True)
             for i in range(int(group.max()) + 1)]
    index = np.concatenate(draws)
    assert len(index) == len(fit_rows) and np.isin(index, fit_rows).all()
    return index


def train(crops, y, rows):
    models = {name: fit(crops[name], y, rows) for name in ARMS}
    models["shared"] = fit(np.vstack([crops[k][rows] for k in ("pre", "post")]), np.tile(y[rows], 2))
    return models


def predict(models, crops):
    value = {name: logits(models[name], crops[name]) for name in ARMS}
    value.update({f"shared_{name}": logits(models["shared"], crops[name]) for name in ("pre", "post")})
    return value


def label_meta(meta, rows, y):
    vector = np.full(len(meta["subject"]), -1, dtype=int)
    vector[rows] = y
    return {**meta, "y": vector}


def combine(base, aux):
    evidence = np.clip(aux - aux.mean(axis=1, keepdims=True), -3, 3)
    return (np.log(base.clip(1e-12)) + WEIGHT * evidence).argmax(1)


def mechanism(meta, rows, y, mode, prediction):
    people = np.array([p.rsplit("/", 1)[-1] for p in meta["subject"][rows]])
    y_meta = label_meta(meta, rows, y)
    result = {}
    for interface in INTERFACES:
        for name in MODES:
            subset = (meta["interface"][rows] == interface) & (mode == name)
            shared = [prediction[f"shared_{arm}"] for arm in ("pre", "post")]
            separate = compare(y_meta, rows[subset], prediction["post"][subset].argmax(1),
                               prediction["pre"][subset].argmax(1))["overall"]["all"]
            by_person = {}
            for person in np.unique(people[subset]):
                at = subset & (people == person) & (y == 1)
                if not at.any():
                    raise ValueError(f"No mental-subtraction rows in {person}/{interface}/{name}")
                pre, post = [100 * float(np.mean(z[at].argmax(1) == 2)) for z in shared]
                odds = [float(np.mean(np.clip(z[at, 2] - z[at, 1], -6, 6))) for z in shared]
                by_person[person] = {"subtraction_windows": int(at.sum()), "pre_sub_to_word": pre,
                                     "post_sub_to_word": post, "post_minus_pre": post - pre,
                                     "post_minus_pre_logodds": odds[1] - odds[0]}
            result[f"{interface}_{name}"] = {"windows": int(subset.sum()), "separate_pre_vs_post": separate,
                                              "participants": by_person}
    cells = [result[f"{i}_{m}"]["participants"] for i in INTERFACES for m in MODES]
    interaction = {}
    for key in ("post_minus_pre", "post_minus_pre_logodds"):
        effects = {p: cells[0][p][key] - cells[1][p][key] - cells[2][p][key] + cells[3][p][key]
                   for p in sorted(set.intersection(*(set(c) for c in cells)))}
        interaction[key] = {"mean": float(np.mean(list(effects.values()))), "people": effects}
    return {"cells": result, "interaction": interaction}


def scores(meta, rows, y, base, aux, masks):
    y_meta = label_meta(meta, rows, y)
    return {name: {"standalone": compare(y_meta, rows, base, aux[name].argmax(1), masks),
                   "overlap_blend": compare(y_meta, rows, base, combine(base, aux[name]), masks)}
            for name in ARMS}


def run(paths, output):
    meta = metadata()
    groups, target, later = partition(meta)
    source = source_rows(meta, "abc")
    assert not np.intersect1d(source, target).size
    n = len(meta["subject"])
    source_mode = modes(meta, source)
    source_y = read_labels(source, n)
    source_x = features(read_eeg(source, n))
    folds = {key: np.isin(source, groups[key]) for key in SINGLES}
    group = np.full(len(source), -1, dtype=int)
    for i, key in enumerate(SINGLES):
        group[folds[key]] = i
    assert (group >= 0).all()
    source_masks = {**folds, **{f"{i}_{m}": (meta["interface"][source] == i) & (source_mode == m)
                               for i in INTERFACES for m in MODES}}
    checksums = [manifest(path) for path in paths]
    proof = [verify_receipts(path, check, seed, meta)
             for path, check, seed in zip(paths, checksums, SEEDS, strict=True)]
    if any(state != proof[0] for state in proof[1:]):
        raise ValueError("Compared seeds use different source geometry")
    report = {"status": "source_complete", "excluded": EVAL, "later_people": LATE, "head_weight": WEIGHT,
              "source": {}, "heldout": {}, "final_source_draws_sha256": {}}
    final = {}
    for seed, path, check in zip(SEEDS, paths, checksums, strict=True):
        oof = {name: np.full((len(source), 3), np.nan) for name in (*ARMS, "shared_pre", "shared_post")}
        for fold, held in enumerate(folds.values()):
            draw = bootstrap(meta, source, source_y, source_mode, np.flatnonzero(~held), seed, fold)
            subset = {name: x[held] for name, x in source_x.items()}
            for name, value in predict(train(source_x, source_y, draw), subset).items():
                oof[name][held] = value
        assert all(np.isfinite(z).all() for z in oof.values())
        base = load_trace(path, check, "source_oof.npz", source, group)["final"]
        report["source"][str(seed)] = {"scores": scores(meta, source, source_y, base, oof, source_masks),
                                       "mechanism": mechanism(meta, source, source_y, source_mode, oof)}
        draw = bootstrap(meta, source, source_y, source_mode, np.arange(len(source)), seed, len(folds))
        report["final_source_draws_sha256"][str(seed)] = hashlib.sha256(source[draw].tobytes()).hexdigest()
        final[seed] = train(source_x, source_y, draw)
    assert len(set(report["final_source_draws_sha256"].values())) == len(SEEDS)
    save(output, report)

    # All candidate heads and fixed contrasts are frozen before held-out EEG or labels are read.
    target_x = features(read_eeg(target, n))
    target_y = read_labels(target, n)
    masks = {"unseen": np.ones(len(target), dtype=bool), "later_unseen": np.isin(target, later)}
    for seed, path, check in zip(SEEDS, paths, checksums, strict=True):
        base = load_trace(path, check, "outer.npz", target)["final"]
        report["heldout"][str(seed)] = scores(meta, target, target_y, base, predict(final[seed], target_x), masks)
    report["status"] = "scored"
    save(output, report)
    print(json.dumps({seed: {arm: m["overlap_blend"]["J_delta"] for arm, m in arms.items()}
                      for seed, arms in report["heldout"].items()}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("primary", "seed100", "seed200", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    run((args.primary, args.seed100, args.seed200), args.output)
