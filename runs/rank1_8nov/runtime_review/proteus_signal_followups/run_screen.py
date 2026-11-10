"""Score one EEG feature bank on source participants against the fixed 10% vote."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

REVIEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW / "proteus_multitaper_wave"))
from spectral import (EVAL, FIXED_WEIGHT, OFFSETS, SCOPES, SINGLES,
                      blend, bootstrap, compare, covariances, diagonal, fit_head,
                      load_trace, manifest, metadata, partition, power_predictions,
                      protocol, read_eeg, read_labels, save, source_rows, summarize,
                      verify_receipts)
from signal_views import hann_geometry, waveform_direction

SEEDS = (400, 500, 600, 700, 800, 900, 1000, 1100)
SCREENS = {
    "geometry": (hann_geometry, {
        "mirror_all": ("power", "rewired3_all", "rewired7_all"),
        "mirror_mid": ("power", "rewired3_mid", "rewired7_mid"),
        "mirror_high": ("power", "rewired3_high", "rewired7_high"),
    }),
    "direction": (waveform_direction, {
        "time_direction": ("even", "random_reversal", "hann_real"),
        "hann_direction": ("hann_even", "hann_random_reversal", "hann_real"),
    }),
}


def source_bank(feature_views):
    meta = metadata()
    groups, reserved, _ = partition(meta)
    rows = source_rows(meta, "abc")
    size = len(meta["subject"])
    people = np.array([s.rsplit("/", 1)[-1] for s in meta["subject"][rows]])
    interfaces = meta["interface"][rows]
    if (len(rows) != 1323 or np.intersect1d(rows, reserved).size or set(people) & set(EVAL)
            or set(meta["session"][rows]) != {"1"}):
        raise ValueError("Unexpected source participant split")
    with np.load(protocol.DATA, allow_pickle=False) as bank:
        channels = bank["ch_names"].tolist()
    raw, y = read_eeg(rows, size), read_labels(rows, size)
    meta["y"] = np.full(size, -1, dtype=int)
    meta["y"][rows] = y
    views = feature_views(raw, channels)
    cov = covariances(raw)
    diag = diagonal(cov)
    power = np.full((len(rows), 3), np.nan)
    fold_ids = np.full(len(rows), -1, dtype=int)
    folds, audit = [], {}
    for i, key in enumerate(SINGLES):
        held = np.isin(rows, groups[key])
        fit_rows = np.flatnonzero(~held)
        if not held.any() or set(people[fit_rows]) & set(people[held]):
            raise ValueError("Source fit and validation participants overlap")
        folds.append((fit_rows, held))
        fold_ids[held] = i
        baseline = fit_head(diag[fit_rows], y[fit_rows])
        power[held] = power_predictions(baseline, cov[held], source_oof=True)
        audit[key] = {"fit_people": np.unique(people[fit_rows]).tolist(),
                      "validation_people": np.unique(people[held]).tolist()}
    return meta, rows, y, people, interfaces, views, power, fold_ids, folds, audit


def fixed_baselines(meta, rows, power, fold_ids, archives, prior):
    reference = json.loads(prior.read_text())["source_blends"]
    bases = {}
    for archive, offset in zip(archives, OFFSETS, strict=True):
        hashes = manifest(archive)
        verify_receipts(archive, hashes, offset, meta)
        overlap = load_trace(archive, hashes, "source_oof.npz", rows, fold_ids)["final"]
        fixed = blend(overlap, power, FIXED_WEIGHT)
        if compare(meta, rows, overlap, fixed) != reference[str(offset)]["diagonal"]:
            raise ValueError("Fixed 10% source comparator mismatch")
        bases[offset] = (overlap, fixed.argmax(1))
    return bases


def summarize_seeds(reports, views, controls):
    by_seed = {arm: {seed: {scope: float(np.mean([v[arm][scope]["delta"] for v in report.values()]))
                          for scope in SCOPES} for seed, report in reports.items()} for arm in views}
    means = {arm: {k: float(np.mean([v[k] for v in scores.values()])) for k in SCOPES}
             for arm, scores in by_seed.items()}
    gates = {}
    for arm, matched in controls.items():
        row = means[arm]
        gains = {control: row["overall"] - means[control]["overall"] for control in matched}
        positive = sum(s["overall"] > 0 for s in by_seed[arm].values())
        checks = {"overall_0p8": row["overall"] >= .8, "brainhero_0p5": row["BrainHero"] >= .5,
                  "graz_nonnegative": row["Graz"] >= 0, "control_0p3": min(gains.values()) >= .3,
                  "six_positive_seeds": positive >= 6}
        gates[arm] = {"pass": all(checks.values()), "checks": checks, "control_gains": gains,
                      "positive_auxiliary_seeds": positive}
    return {"mean_deltas": means, "by_seed": by_seed, "source_gate": gates}


def run(screen, archives, prior, output):
    feature_views, controls = SCREENS[screen]
    meta, rows, y, people, interfaces, views, power, fold_ids, folds, audit = source_bank(feature_views)
    bases = fixed_baselines(meta, rows, power, fold_ids, archives, prior)
    reports = {}
    for seed in SEEDS:
        probs = {arm: np.full_like(power, np.nan) for arm in views}
        for i, (fit_rows, held) in enumerate(folds):
            draw = bootstrap(fit_rows, y, people, interfaces, seed, i)
            for arm, view in views.items():
                model = fit_head(view[draw], y[draw])
                probs[arm][held] = softmax(model.decision_function(view[held]), axis=1)
        if not all(np.isfinite(v).all() for v in probs.values()):
            raise ValueError("Missing source predictions")
        reports[seed] = {offset: {arm: summarize(meta, rows, fixed,
                            blend(overlap, prob, FIXED_WEIGHT).argmax(1))
                            for arm, prob in probs.items()} for offset, (overlap, fixed) in bases.items()}
    summary = summarize_seeds(reports, views, controls)
    save(output, {"screen": screen, "excluded": EVAL, "source_windows": len(rows), "audit": audit,
                  "scores": reports, **summary})
    print(json.dumps({"means": summary["mean_deltas"], "gates": summary["source_gate"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("screen", choices=SCREENS)
    p.add_argument("archives", nargs=3, type=Path)
    p.add_argument("prior", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.screen, args.archives, args.prior, args.output)
