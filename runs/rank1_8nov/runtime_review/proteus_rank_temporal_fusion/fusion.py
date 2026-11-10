"""Source-only rank covariance and bounded lag energy comparison."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

REVIEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW / "proteus_signal_followups"))
from run_screen import fixed_baselines, source_bank, summarize_seeds
from signal_views import waveform_direction
from spectral import EVAL, FIXED_WEIGHT, blend, bootstrap, fit_head, save, summarize
from views import copula_views, tangent_views

SEEDS = (2000, 2100, 2200, 2300)
CONTROLS = {"rank_fusion": ("linear_fusion", "rank_covariance", "hann_lag")}


def feature_bank(raw, channels):
    return copula_views(raw), waveform_direction(raw, channels)["hann_even"]


def probabilities(covs, temporal, folds, labels, people, interfaces, seed):
    names = ("rank_covariance", "linear_covariance", "hann_lag")
    probs = {name: np.full((len(labels), 3), np.nan) for name in names}
    for i, (fit_rows, held) in enumerate(folds):
        draw = bootstrap(fit_rows, labels, people, interfaces, seed, i)
        spatial = tangent_views(covs, draw)
        for name, view in zip(names, (spatial["copula"], spatial["linear"], temporal), strict=True):
            head = fit_head(view[draw], labels[draw])
            probs[name][held] = softmax(head.decision_function(view[held]), axis=1)
    if not all(np.isfinite(prob).all() for prob in probs.values()):
        raise ValueError("Incomplete source probabilities")
    probs["rank_fusion"] = (probs["rank_covariance"] + probs["hann_lag"]) * .5
    probs["linear_fusion"] = (probs["linear_covariance"] + probs["hann_lag"]) * .5
    return probs


def report(meta, rows, power, bases, probs, add):
    result = {}
    for offset, (overlap, fixed_labels) in bases.items():
        base = blend(overlap, power, FIXED_WEIGHT) if add else overlap
        result[offset] = {arm: summarize(meta, rows, fixed_labels,
                          blend(base, probability, FIXED_WEIGHT).argmax(1))
                          for arm, probability in probs.items()}
    return result


def run(archives, prior, output):
    meta, rows, y, people, interfaces, (covs, temporal), power, fold_ids, folds, audit = source_bank(feature_bank)
    bases = fixed_baselines(meta, rows, power, fold_ids, archives, prior)
    reports = {"replacement": {}, "additional": {}}
    for seed in SEEDS:
        probs = probabilities(covs, temporal, folds, y, people, interfaces, seed)
        reports["replacement"][seed] = report(meta, rows, power, bases, probs, add=False)
        reports["additional"][seed] = report(meta, rows, power, bases, probs, add=True)
        print(f"Completed source auxiliary seed {seed}", flush=True)
    summary = {mode: summarize_seeds(values, probs, CONTROLS if mode == "replacement" else {})
               for mode, values in reports.items()}
    # The shared screen used eight seeds; this protocol fixes three of four.
    gate = summary["replacement"]["source_gate"]["rank_fusion"]
    del gate["checks"]["six_positive_seeds"]
    gate["checks"]["three_positive_seeds"] = gate["positive_auxiliary_seeds"] >= 3
    gate["pass"] = all(gate["checks"].values())
    save(output, {"source_windows": len(rows), "audit": audit, "protected_exclusions": EVAL,
                  "summary": summary, "scores": reports})
    print(json.dumps({mode: value["mean_deltas"] for mode, value in summary.items()}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archives", nargs=3, type=Path)
    p.add_argument("prior", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.archives, a.prior, a.output)
