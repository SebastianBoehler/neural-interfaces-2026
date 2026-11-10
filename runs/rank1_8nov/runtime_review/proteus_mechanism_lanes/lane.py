"""Run matched source-only mechanism experiments against the fixed 10% baseline."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

REVIEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW / "proteus_cognitive_csp_wave"))
from screen_csp import (EVAL, FIXED_WEIGHT, OFFSETS, SINGLES, blend, compare, covariances, diagonal,
                        fit_head, load_trace, manifest, metadata, partition, power_predictions, read_eeg,
                        read_labels, save, source_rows, verify_receipts)

sys.path.insert(0, str(REVIEW / "proteus_one_way_gate_wave"))
from gate import official_section
from views import copula_views, nuisance_views, ordered_views, tangent_views

CONTRASTS = {"copula": ("copula", "linear"),
             "nuisance": ("nuisance_removed", "placebo_removed"),
             "time": ("ordered", "shuffled_order")}
WEIGHT = .10


def bootstrap(rows, y, people, interface, seed, fold):
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
    cells = [rows[(people[rows] == p) & (interface[rows] == k) & (y[rows] == c)]
             for p in np.unique(people[rows]) for k in ("BrainHero", "Graz") for c in range(3)]
    if any(len(x) < 2 for x in cells):
        raise ValueError("Missing class coverage in source bootstrap")
    return np.concatenate([rng.choice(x, len(x), replace=True) for x in cells])


def collect(lane, raw, diagonal_features):
    if lane == "copula":
        return copula_views(raw)
    if lane == "time":
        return ordered_views(raw)
    return diagonal_features


def predict(lane, feature_bank, draw, held, y, person, interface, seed, fold):
    if lane == "copula":
        views = tangent_views(feature_bank, draw)
    elif lane == "nuisance":
        views = nuisance_views(feature_bank, draw, y, person, interface, seed, fold)
    else:
        views = feature_bank
    logits = {}
    for key, x in views.items():
        if key == "linear_compressed":
            continue
        model = fit_head(x[draw], y[draw])
        logits[key] = model.decision_function(x[held]).astype(np.float64)
        if lane == "copula" and key == "linear":
            logits["linear_compressed"] = model.decision_function(views["linear_compressed"][held]).astype(np.float64)
    return logits


def summarize(meta, rows, base, candidate):
    fixed = official_section(meta, rows, base, candidate)
    return {k: {"score": q["candidate"], "delta": q["delta"], "switches": q["switches"]}
            for k, q in fixed.items()}


def run(lane, paths, prior, output):
    meta = metadata()
    groups, excluded, _ = partition(meta)
    rows = source_rows(meta, "abc")
    people = np.array([p.rsplit("/", 1)[-1] for p in meta["subject"][rows]])
    interface = meta["interface"][rows]
    if (len(rows) != 1323 or np.intersect1d(rows, excluded).size or set(people) & set(EVAL)
            or set(meta["session"][rows]) != {"1"}):
        raise ValueError("Invalid source-only cohort")
    raw = read_eeg(rows, len(meta["subject"]))
    y = read_labels(rows, len(meta["subject"]))
    meta["y"] = np.full(len(meta["subject"]), -1, dtype=int)
    meta["y"][rows] = y
    cov = covariances(raw)
    diag = diagonal(cov)
    bank = collect(lane, raw, diag)
    folds = {k: np.isin(rows, groups[k]) for k in SINGLES}
    fold_id = np.full(len(rows), -1, dtype=int)
    power = np.full((len(rows), 3), np.nan)
    for i, held in enumerate(folds.values()):
        if set(people[held]) & set(people[~held]):
            raise ValueError("Source person overlap")
        fold_id[held] = i
        model = fit_head(diag[~held], y[~held])
        power[held] = power_predictions(model, cov[held], source_oof=True)
    old = json.loads(prior.read_text())["source_blends"]
    audits, reports = {}, {}
    for path, seed in zip(paths, OFFSETS, strict=True):
        check = manifest(path)
        verify_receipts(path, check, seed, meta)
        oof = load_trace(path, check, "source_oof.npz", rows, fold_id)["final"]
        base = blend(oof, power, FIXED_WEIGHT)
        if compare(meta, rows, oof, base) != old[str(seed)]["diagonal"]:
            raise ValueError("Fixed 10% source comparator mismatch")
        base = base.astype(np.float64)
        predictions, audit = {}, {}
        for i, (key, held) in enumerate(folds.items()):
            draw = bootstrap(np.flatnonzero(~held), y, people, interface, seed, i)
            data = predict(lane, bank, draw, held, y, people, interface, seed, i)
            for name, logits in data.items():
                predictions.setdefault(name, np.full((len(rows), 3), np.nan))[held] = logits
            audit[key] = {"fit_people": np.unique(people[draw]).tolist(),
                          "validation_people": np.unique(people[held]).tolist(),
                          "bootstrap_sha256": hashlib.sha256(draw.tobytes()).hexdigest()}
        fixed = base.argmax(1)
        scores = {}
        for name, logits in predictions.items():
            if not np.isfinite(logits).all():
                raise ValueError("Missing out-of-fold predictions")
            aux = softmax(logits, axis=1)
            scores[name] = {"blend_vs_fixed": summarize(meta, rows, fixed, blend(base, aux, WEIGHT).argmax(1)),
                            "standalone_vs_fixed": summarize(meta, rows, fixed, aux.argmax(1))}
        audits[str(seed)], reports[str(seed)] = audit, scores
        print(f"Completed source-only seed {seed} for {lane}", flush=True)
    candidate, control = CONTRASTS[lane]
    means = {arm: {scope: float(np.mean([s[arm]["blend_vs_fixed"][scope]["delta"] for s in reports.values()]))
                   for scope in ("overall", "BrainHero", "Graz")} for arm in predictions}
    change = means[candidate]
    criteria = {"overall_0p8": change["overall"] >= .8, "brainhero_0p5": change["BrainHero"] >= .5,
                "graz_nonnegative": change["Graz"] >= -1e-12,
                "beats_control_0p3": change["overall"] - means[control]["overall"] >= .3,
                "two_positive_seeds": sum(s[candidate]["blend_vs_fixed"]["overall"]["delta"] > 0
                                          for s in reports.values()) >= 2}
    save(output, {"lane": lane, "source_windows": len(rows), "excluded": EVAL, "fit_and_validation": audits,
                  "seed_scores": reports, "mean_deltas": means, "candidate": candidate, "control": control,
                  "source_gate": {"pass": all(criteria.values()), "checks": criteria}})
    print(json.dumps({"mean_deltas": means, "checks": criteria, "pass": all(criteria.values())}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lane", choices=CONTRASTS)
    p.add_argument("archives", nargs=3, type=Path)
    p.add_argument("previous_source_fit", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.lane, a.archives, a.previous_source_fit, a.output)
