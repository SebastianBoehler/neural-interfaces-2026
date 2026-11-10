"""Run one frozen overnight method and auxiliary seed on PROTEUS source folds."""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

REVIEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW / "proteus_breadth_48"))
from breadth_run import EVAL, FIXED_WEIGHT, blend, bootstrap, save, summarize
from run_screen import fixed_baselines, source_bank

from readouts import markov, ordinary, pairwise, prototypes
from signal_banks import covariance_banks, cross_view, dense_spectral, half_window, prototypes as prototype_bank, tile_bank
from small_models import bilinear, patch

SEEDS = (4200, 4300, 4400, 4500, 4600, 4700)
METHODS = {
    "bounded_covariance": (covariance_banks, ordinary, "bounded", ("linear", "rank", "power")),
    "stable_lag": (half_window, ordinary, "geometric", ("arithmetic", "full", "hann")),
    "dense_spectral": (dense_spectral, ordinary, "spectral", ("linear", "rank", "power")),
    "pairwise": (cross_view, pairwise, "temporal_pair", ("temporal_multi", "rank_multi")),
    "prototypes": (prototype_bank, prototypes, "rank_template", ("linear_template", "rank_logistic")),
    "markov": (tile_bank, markov, "ordered", ("bag", "shuffled", "pooled")),
    "patch": (tile_bank, patch, "ordered", ("shuffled", "pooled")),
    "bilinear": (cross_view, bilinear, "product", ("additive", "shuffled", "temporal")),
}


def run(slot, archives, prior, folder):
    method_index, seed_index = divmod(slot, len(SEEDS))
    method, seed = tuple(METHODS)[method_index], SEEDS[seed_index]
    feature_fn, readout, candidate, controls = METHODS[method]
    meta, rows, y, people, interfaces, bank, power, fold_ids, folds, audit = source_bank(feature_fn)
    bases = fixed_baselines(meta, rows, power, fold_ids, archives, prior)
    probs, draw_hashes = {}, []
    for i, (fit_rows, held) in enumerate(folds):
        draw = bootstrap(fit_rows, y, people, interfaces, seed, i)
        draw_hashes.append(hashlib.sha256(draw.tobytes()).hexdigest())
        forecasts = readout(bank, draw, held, y, people, interfaces, seed, i)
        for key, value in forecasts.items():
            probs.setdefault(key, np.full_like(power, np.nan))[held] = value
    if not {candidate, *controls} <= probs.keys() or not all(
        np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1) for p in probs.values()
    ):
        raise ValueError("Incomplete source probability bank")
    scores = {offset: {name: summarize(meta, rows, fixed,
                       blend(overlap, p, FIXED_WEIGHT).argmax(1))
                       for name, p in probs.items()} for offset, (overlap, fixed) in bases.items()}
    save(folder / f"{method}-{seed}.json",
         {"method": method, "auxiliary_seed": seed, "source_windows": len(rows),
          "protected_exclusions": EVAL, "audit": audit, "draw_sha256": draw_hashes,
          "candidate": candidate, "controls": controls, "scores": scores})
    print(f"Completed {method} auxiliary seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slot", type=int, choices=range(len(METHODS) * len(SEEDS)))
    parser.add_argument("archives", nargs=3, type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    run(args.slot, args.archives, args.prior, args.folder)
