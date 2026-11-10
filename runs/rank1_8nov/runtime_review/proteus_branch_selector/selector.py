"""Test nested source-only selection between mid-band and Hann/lag branches."""

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REVIEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW / "proteus_midband_fusion"))
from mid_fusion import feature_bank, EVAL, FIXED_WEIGHT, blend, bootstrap, fit_head, save, summarize
from run_screen import fixed_baselines, source_bank, summarize_seeds

SEEDS = (3600, 3700, 3800, 3900, 4000, 4100)
CONTROLS = {"selector": ("equal", "shuffled", "mid_rank", "hann_lag")}
BRANCHES = ("mid_rank", "hann_lag")


def predict(views, train, held, y):
    return np.stack([softmax(fit_head(views[name][train], y[train]).decision_function(views[name][held]), axis=1)
                     for name in BRANCHES], axis=1)


def describe(p):
    ranked = np.sort(p, axis=-1)
    return np.column_stack((p.reshape(len(p), -1), ranked[..., -1] - ranked[..., -2]))


def selection_weights(p):
    # Confidence bounds fixed before training. Abstention retains the equal blend.
    return np.where(p >= .65, 1.0, np.where(p <= .35, 0.0, .5))


def selector_fit(x, wins, eligible):
    if min(np.bincount(wins[eligible], minlength=2)) < 5:
        raise ValueError("Not enough decisive source disagreements")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))
    return model.fit(x[eligible], wins[eligible])


def gate_fold(views, y, people, interfaces, fold_ids, train, held, seed, fold):
    inner_p, inner_rows = [], []
    for inner in np.unique(fold_ids[train]):
        fit = train[fold_ids[train] != inner]
        valid = train[fold_ids[train] == inner]
        if set(people[fit]) & (set(people[valid]) | set(people[held])):
            raise ValueError("Nested source participants overlap")
        # Inner single-group fits avoid bootstrap restrictions on small classes.
        inner_p.append(predict(views, fit, valid, y))
        inner_rows.append(valid)
    p = np.concatenate(inner_p)
    truth = y[np.concatenate(inner_rows)]
    correct = p.argmax(axis=2) == truth[:, None]
    eligible = correct[:, 0] != correct[:, 1]
    wins = correct[:, 0].astype(int)
    features = describe(p)
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 7200]))
    shuffled = wins.copy()
    shuffled[eligible] = rng.permutation(wins[eligible])
    trained = selector_fit(features, wins, eligible)
    control = selector_fit(features, shuffled, eligible)
    draw = bootstrap(train, y, people, interfaces, seed, fold)
    forecast = predict(views, draw, held, y)
    x = describe(forecast)
    rank, hann = forecast[:, 0], forecast[:, 1]
    equal = .5 * (rank + hann)
    decisions = {"mid_rank": rank, "hann_lag": hann, "equal": equal}
    for key, gate in (("selector", trained), ("shuffled", control)):
        weight = selection_weights(gate.predict_proba(x)[:, 1])[:, None]
        decisions[key] = rank * weight + hann * (1 - weight)
    # Labelled interface is an upper-bound diagnostic and never enters the selector.
    decisions["interface_oracle"] = np.where((interfaces[held] == "BrainHero")[:, None], rank, hann)
    return decisions, int(eligible.sum())


def run(archives, prior, output):
    meta, rows, y, people, interfaces, views, power, fold_ids, folds, audit = source_bank(feature_bank)
    bases = fixed_baselines(meta, rows, power, fold_ids, archives, prior)
    reports, eligible = {}, {}
    for seed in SEEDS:
        probs, eligible[seed] = {}, []
        for i, (train, held) in enumerate(folds):
            forecasts, count = gate_fold(views, y, people, interfaces, fold_ids, train, held, seed, i)
            eligible[seed].append(count)
            for key, value in forecasts.items():
                probs.setdefault(key, np.full_like(power, np.nan))[held] = value
        if not all(np.isfinite(v).all() for v in probs.values()):
            raise ValueError("Incomplete nested source probabilities")
        reports[seed] = {offset: {arm: summarize(meta, rows, fixed,
                         blend(overlap, probability, FIXED_WEIGHT).argmax(1))
                         for arm, probability in probs.items()} for offset, (overlap, fixed) in bases.items()}
    summary = summarize_seeds(reports, probs, CONTROLS)
    gate = summary["source_gate"]["selector"]
    del gate["checks"]["six_positive_seeds"]
    gate["checks"]["five_positive_seeds"] = gate["positive_auxiliary_seeds"] >= 5
    gate["pass"] = all(gate["checks"].values())
    save(output, {"source_windows": len(rows), "protected_exclusions": EVAL, "audit": audit,
                  "eligible_inner_rows": eligible, "summary": summary, "scores": reports})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archives", nargs=3, type=Path)
    p.add_argument("prior", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.archives, a.prior, a.output)
