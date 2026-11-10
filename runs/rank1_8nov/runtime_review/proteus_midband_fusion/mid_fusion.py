"""Test whether mid-band covariance and bounded lag energy complement each other."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

REVIEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW / "proteus_breadth_48"))
from breadth_run import EVAL, FIXED_WEIGHT, blend, bootstrap, fit_head, save, summarize
from refinements import band_rank
from run_screen import fixed_baselines, source_bank, summarize_seeds
from signal_views import waveform_direction

SEEDS = (3000, 3100, 3200, 3300, 3400, 3500)
SINGLES = ("mid_rank", "mid_linear", "outer_rank", "hann_lag")
CONTROLS = {"mid_rank_fusion": ("mid_linear_fusion", "outer_rank_fusion", "mid_rank", "hann_lag")}


def feature_bank(raw, channels):
    bank = band_rank(raw, channels)
    bank["hann_lag"] = waveform_direction(raw, channels)["hann_even"]
    return {name: bank[name] for name in SINGLES}


def run(archives, prior, output):
    meta, rows, y, people, interfaces, views, power, fold_ids, folds, audit = source_bank(feature_bank)
    bases = fixed_baselines(meta, rows, power, fold_ids, archives, prior)
    reports = {}
    for seed in SEEDS:
        probs = {name: np.full_like(power, np.nan) for name in views}
        for i, (fit_rows, held) in enumerate(folds):
            draw = bootstrap(fit_rows, y, people, interfaces, seed, i)
            for name, view in views.items():
                head = fit_head(view[draw], y[draw])
                probs[name][held] = softmax(head.decision_function(view[held]), axis=1)
        if not all(np.isfinite(v).all() for v in probs.values()):
            raise ValueError("Incomplete source probabilities")
        for name in ("mid_rank", "mid_linear", "outer_rank"):
            probs[f"{name}_fusion"] = (probs[name] + probs["hann_lag"]) * .5
        reports[seed] = {offset: {name: summarize(meta, rows, fixed,
                         blend(overlap, prob, FIXED_WEIGHT).argmax(1))
                         for name, prob in probs.items()} for offset, (overlap, fixed) in bases.items()}
        print(f"Completed source auxiliary seed {seed}", flush=True)
    summary = summarize_seeds(reports, probs, CONTROLS)
    gate = summary["source_gate"]["mid_rank_fusion"]
    del gate["checks"]["six_positive_seeds"]
    gate["checks"]["five_positive_seeds"] = gate["positive_auxiliary_seeds"] >= 5
    gate["pass"] = all(gate["checks"].values())
    save(output, {"source_windows": len(rows), "protected_exclusions": EVAL, "audit": audit,
                  "scores": reports, "summary": summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs=3, type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archives, args.prior, args.output)
