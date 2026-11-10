"""Run one prespecified EEG method and bootstrap draw on source folds."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

REVIEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW / "proteus_signal_followups"))
from run_screen import fixed_baselines, source_bank
from spectral import EVAL, FIXED_WEIGHT, blend, bootstrap, fit_head, save, summarize

from new_methods import bispectrum, interaction_bank, interaction_forecasts, ordinal, spatial_modes
from refinements import band_rank, compact_rank, hemispheres, lag_contrast

SEEDS = (2400, 2500, 2600, 2700, 2800, 2900)
METHODS = {
    "compact_rank": (compact_rank, "rank", ("linear", "hann_lag")),
    "lag_contrast": (lag_contrast, "relative", ("plain", "log_energy", "shuffled", "hann")),
    "hemispheres": (hemispheres, "same_side", ("mirror", "rewired", "power")),
    "band_rank": (band_rank, "mid_rank", ("mid_linear", "outer_rank", "outer_linear", "power")),
    "ordinal": (ordinal, "ordinal", ("phase_scrambled", "hann_lag")),
    "bispectrum": (bispectrum, "bispectrum", ("phase_scrambled", "hann")),
    "spatial_modes": (spatial_modes, "eigenmodes", ("rewired", "eigenvalues", "hann")),
    "interactions": (interaction_bank, "nonlinear", ("linear_projection", "hann_lag")),
}


def run(method, seed, archives, prior, output):
    features, candidate, controls = METHODS[method]
    meta, rows, y, people, interfaces, views, power, fold_ids, folds, audit = source_bank(features)
    bases = fixed_baselines(meta, rows, power, fold_ids, archives, prior)
    probs, draw_hashes = {}, []
    for i, (fit_rows, held) in enumerate(folds):
        draw = bootstrap(fit_rows, y, people, interfaces, seed, i)
        draw_hashes.append(hashlib.sha256(draw.tobytes()).hexdigest())
        if method == "interactions":
            forecasts = interaction_forecasts(views, draw, held, y, seed, i)
        else:
            forecasts = {arm: softmax(fit_head(x[draw], y[draw]).decision_function(x[held]), axis=1)
                         for arm, x in views.items()}
        for name, values in forecasts.items():
            probs.setdefault(name, np.full_like(power, np.nan))[held] = values
    if not all(np.isfinite(v).all() for v in probs.values()):
        raise ValueError("Incomplete source predictions")
    if not {candidate, *controls} <= probs.keys():
        raise ValueError("Candidate or matched control is missing")
    reports = {offset: {arm: summarize(meta, rows, fixed,
                 blend(overlap, probability, FIXED_WEIGHT).argmax(1))
                 for arm, probability in probs.items()} for offset, (overlap, fixed) in bases.items()}
    save(output, {"method": method, "auxiliary_seed": seed, "source_windows": len(rows),
                  "protected_exclusions": EVAL, "audit": audit, "draw_sha256": draw_hashes,
                  "candidate": candidate, "controls": controls, "scores": reports})
    print(f"Completed {method} auxiliary seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slot", type=int, choices=range(len(METHODS) * len(SEEDS)))
    parser.add_argument("archives", nargs=3, type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("job")
    args = parser.parse_args()
    index, seed_index = divmod(args.slot, len(SEEDS))
    method, seed = tuple(METHODS)[index], SEEDS[seed_index]
    run(method, seed, args.archives, args.prior, args.directory / f"{method}-{seed}-{args.job}.json")
