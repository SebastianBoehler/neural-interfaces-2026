"""Fit a shared channel-power blend weight on the archived PROTEUS source folds."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REVIEW = Path(__file__).resolve().parents[1] / "runs/rank1_8nov/runtime_review"
sys.path.insert(0, str(REVIEW / "proteus_interface_wave"))
from screen import covariances
from run import EVAL, SINGLES, metadata, partition, read_eeg, read_labels, save, source_rows
from confirm_seeds import OFFSETS, load_trace, manifest, verify_receipts
from score_candidate import compare

FIXED_WEIGHT = 0.10
MAX_WEIGHT = 0.20


def channel_power(eeg):
    cov = covariances(eeg)
    return np.log(cov.diagonal(axis1=-2, axis2=-1)).reshape(len(cov), -1)


def fit_head(x, labels):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=500))
    model.fit(x, labels)
    classifier = model[-1]
    if classifier.classes_.tolist() != [0, 1, 2] or classifier.n_iter_.max() >= classifier.max_iter:
        raise ValueError("Invalid fitted channel-power classifier")
    return model


def model_sha256(model):
    scaler, head = model
    return hashlib.sha256(b"".join(x.tobytes() for x in
                                  (scaler.mean_, scaler.scale_, head.coef_, head.intercept_, head.classes_))).hexdigest()


def blend(base, auxiliary, weight):
    if weight == 0:
        return base.copy()
    return (1 - weight) * base + weight * auxiliary


def fit_weight(bases, auxiliary, labels, people, interfaces):
    """Minimize person/interface/class-balanced out-of-fold log loss."""
    if bases.shape != (len(OFFSETS), len(labels), 3) or auxiliary.shape != (len(labels), 3):
        raise ValueError("Expected three aligned source overlap posteriors and one auxiliary posterior")
    if any(not np.isfinite(p).all() or (p < 0).any()
           or not np.allclose(p.sum(axis=-1), 1, atol=1e-7, rtol=0)
           for p in (bases, auxiliary)):
        raise ValueError("Source predictions must be finite normalized probabilities")
    _, cell, counts = np.unique(np.column_stack((people, interfaces, labels)), axis=0,
                                return_inverse=True, return_counts=True)
    if len(counts) != len(np.unique(people)) * 2 * 3:
        raise ValueError("Expected both interfaces and all three labels for each source person")
    mass = 1 / (len(OFFSETS) * len(counts) * counts[cell])
    index = np.arange(len(labels))
    p = np.maximum(bases[:, index, labels], 1e-12)
    direction = np.maximum(auxiliary[index, labels], 1e-12) - p

    def loss(w):
        return float(-np.sum(mass * np.log(p + w * direction)))

    def gradient(w):
        return float(-np.sum(mass * direction / (p + w * direction)))

    if gradient(0.0) >= 0:
        weight = 0.0
    elif gradient(MAX_WEIGHT) <= 0:
        weight = MAX_WEIGHT
    else:
        weight = float(brentq(gradient, 0, MAX_WEIGHT))
    return {"weight": weight, "bounds": [0.0, MAX_WEIGHT],
            "log_loss": {"overlap": loss(0.0), "fixed_10pct": loss(FIXED_WEIGHT), "source_fit": loss(weight)},
            "derivative_at_bounds": [gradient(0.0), gradient(MAX_WEIGHT)]}


def compare_arms(meta, rows, base, auxiliary, weight, slices=None):
    fixed = blend(base, auxiliary, FIXED_WEIGHT)
    fitted = blend(base, auxiliary, weight)
    return {
        "fixed_vs_overlap": compare(meta, rows, base, fixed, slices),
        "fitted_vs_overlap": compare(meta, rows, base, fitted, slices),
        "fitted_vs_fixed": compare(meta, rows, fixed, fitted, slices),
    }


def run(paths, fixed_fit, fixed_result, fit_path, result_path):
    meta = metadata()
    groups, target, later = partition(meta)
    source = source_rows(meta, "abc")
    people = np.array([p.rsplit("/", 1)[-1] for p in meta["subject"][source]])
    if (len(source) != 1323 or len(target) != 1332 or len(later) != 762
            or set(people) & set(EVAL) or np.intersect1d(source, target).size
            or set(meta["session"][source]) != {"1"}):
        raise ValueError("PROTEUS source and evaluation split differ from the prior replay")

    manifests = [manifest(path) for path in paths]
    provenance = [verify_receipts(path, checks, seed, meta)
                  for path, checks, seed in zip(paths, manifests, OFFSETS, strict=True)]
    if any(proof != provenance[0] for proof in provenance[1:]):
        raise ValueError("Overlap seeds have different source geometry")
    labels = read_labels(source, len(meta["subject"]))
    features = channel_power(read_eeg(source, len(meta["subject"])))
    meta["y"] = np.full(len(meta["subject"]), -1, dtype=int)
    meta["y"][source] = labels
    fold_id = np.full(len(source), -1, dtype=int)
    oof = np.full((len(source), 3), np.nan)
    folds = {}
    for i, key in enumerate(SINGLES):
        held = np.isin(source, groups[key])
        fit_people, held_people = np.unique(people[~held]), np.unique(people[held])
        if not held.any() or np.intersect1d(fit_people, held_people).size:
            raise ValueError("Source fit and source validation must be disjoint by participant")
        model = fit_head(features[~held], labels[~held])
        # The previous replay assembled float64 out-of-fold logits before softmax.
        oof[held] = softmax(model.decision_function(features[held]).astype(np.float64), axis=1)
        fold_id[held] = i
        folds[key] = {"fit_people": fit_people.tolist(), "validation_people": held_people.tolist(),
                      "head_sha256": model_sha256(model)}
    bases = np.stack([load_trace(path, check, "source_oof.npz", source, fold_id)["final"]
                      for path, check in zip(paths, manifests, strict=True)])
    selection = fit_weight(bases, oof, labels, people, meta["interface"][source])
    weight = selection["weight"]
    source_scores = {str(seed): compare_arms(meta, source, base, oof, weight)
                     for seed, base in zip(OFFSETS, bases, strict=True)}
    old_source = json.loads(fixed_fit.read_text())
    for seed, value in source_scores.items():
        if value["fixed_vs_overlap"] != old_source["source_blends"][seed]["diagonal"]:
            raise ValueError("Fixed 10% blend failed source replay parity")
    model = fit_head(features, labels)
    receipt = {"excluded_from_fitting_and_tuning": EVAL, "source_people": np.unique(people).tolist(),
               "source_windows": len(source), "source_sessions": ["1"], "folds": folds,
               "head_sha256": model_sha256(model), "fixed_weight": FIXED_WEIGHT,
               "selection": selection, "source_scores": source_scores}
    save(fit_path, receipt)
    print(f"Froze source weight {weight:.9f} and head before reserved inference", flush=True)

    # The frozen head sees reserved EEG for inference. Labels only enter scoring.
    test_features = channel_power(read_eeg(target, len(meta["subject"])))
    test_power = softmax(model.decision_function(test_features), axis=1)
    if model_sha256(model) != receipt["head_sha256"]:
        raise ValueError("Head changed during frozen inference")
    test_base = [load_trace(path, check, "outer.npz", target)["final"]
                 for path, check in zip(paths, manifests, strict=True)]
    meta["y"][target] = read_labels(target, len(meta["subject"]))
    slices = {"unseen": np.ones(len(target), dtype=bool), "later_unseen": np.isin(target, later)}
    scores = {str(seed): compare_arms(meta, target, base, test_power, weight, slices)
              for seed, base in zip(OFFSETS, test_base, strict=True)}
    old_outer = json.loads(fixed_result.read_text())
    for seed, value in scores.items():
        if value["fixed_vs_overlap"] != old_outer["seed_scores"][seed]["diagonal"]:
            raise ValueError("Fixed 10% blend failed reserved replay parity")
    save(result_path, {"fit_receipt_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                       "weights": {"fixed": FIXED_WEIGHT, "source_fit": weight}, "scores": scores,
                       "excluded_from_fitting_and_tuning": EVAL,
                       "target_windows": {name: int(at.sum()) for name, at in slices.items()},
                       "outer_trace_sha256": {str(seed): check["outer.npz"]
                                              for seed, check in zip(OFFSETS, manifests, strict=True)}})
    print(json.dumps({s: {a: d["J_delta"] for a, d in arms.items()} for s, arms in scores.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs=3)
    for name in ("fixed_fit", "fixed_result", "fit_receipt", "result"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.archives, args.fixed_fit, args.fixed_result, args.fit_receipt, args.result)
