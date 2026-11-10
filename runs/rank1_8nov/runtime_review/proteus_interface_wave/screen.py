"""Source-only PROTEUS screens on frozen participant folds."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / wave) for wave in ("feedback_crop_wave", "brainhero_seed_wave")]
from experiment import modes
from run import EVAL, SINGLES, metadata, partition, read_eeg, read_labels, save, source_rows
from experiments.features import expm_sym, logm, whiten_tangent

BANDS = ((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))
SCOPES = ("overall", "BrainHero", "Graz")


def covariances(x):
    f = np.fft.rfftfreq(x.shape[-1], 1 / 120)
    taper = np.hanning(x.shape[-1])
    eye = np.eye(x.shape[1])
    result = np.empty((len(x), len(BANDS), len(eye), len(eye)), dtype=np.float32)
    for start in range(0, len(x), 64):
        rows = slice(start, start + 64)
        chunk = x[rows]
        spec = np.fft.rfft((chunk - chunk.mean(axis=-1, keepdims=True)) * taper, axis=-1)
        for b, (lo, hi) in enumerate(BANDS):
            power = spec[..., (f >= lo) & (f < hi)]
            c = (power @ power.swapaxes(-2, -1).conj()).real
            trace = np.trace(c, axis1=-2, axis2=-1)
            if not np.all(np.isfinite(trace) & (trace > 0)):
                raise ValueError("Nonfinite or empty PROTEUS band covariance")
            result[rows, b] = 0.95 * c * (len(eye) / trace)[:, None, None] + 0.05 * eye
    return result


def geometry(cov, fit, val, y):
    diag = np.log(cov.diagonal(axis1=-2, axis2=-1)).reshape(len(cov), -1)
    parts = []
    for b in range(len(BANDS)):
        band = cov[:, b].astype(np.float64)
        reference = expm_sym(logm(band[fit]).mean(axis=0))
        parts.append(whiten_tangent(band, reference))
    tangent = np.column_stack(parts)
    logits = {}
    for arm, features in (("diagonal", diag), ("tangent", tangent)):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=500))
        clf.fit(features[fit], y[fit])
        if clf[-1].n_iter_.max() >= clf[-1].max_iter:
            raise RuntimeError(f"{arm} logistic regression did not converge")
        logits[arm] = clf.decision_function(features[val])
    return logits


def score(logits, y, people, interface):
    if logits.shape != (len(y), 3) or not np.isfinite(logits).all():
        raise ValueError("Expected three finite scores per trial")
    pred = logits.argmax(axis=1)
    cognitive = logits[:, 1:].argmax(axis=1) + 1
    results = {}
    for scope in SCOPES:
        cells = {}
        for person in np.unique(people):
            for context in np.unique(interface) if scope == "overall" else (scope,):
                at = (people == person) & (interface == context)
                if set(y[at]) != {0, 1, 2}:
                    raise ValueError(f"Missing class for {person}/{context}")
                cells[f"{person}/{context}"] = {
                    "ba": 100 * float(np.mean([np.mean(pred[at & (y == k)] == k) for k in range(3)])),
                    "subtraction_vs_word_ba": 100 * float(np.mean(
                        [np.mean(cognitive[at & (y == k)] == k) for k in (1, 2)])),
                }
        results[scope] = {key: float(np.mean([cell[key] for cell in cells.values()]))
                          for key in ("ba", "subtraction_vs_word_ba")}
        results[scope]["cells"] = cells
    return results


def run(lane, seed, output):
    meta = metadata()
    rows = source_rows(meta, "abc")
    groups, excluded_rows, _ = partition(meta)
    people = np.array([p.rsplit("/", 1)[-1] for p in meta["subject"][rows]])
    if (len(rows) != 1323 or np.intersect1d(rows, excluded_rows).size or set(people) & set(EVAL)
            or set(meta["session"][rows]) != {"1"}):
        raise ValueError("PROTEUS source isolation failed")
    y = read_labels(rows, len(meta["subject"]))
    interface, mode = meta["interface"][rows], modes(meta, rows)
    cov = covariances(read_eeg(rows, len(meta["subject"])))
    if lane == "paired":
        import torch
        from paired import fit_predict
        if not torch.cuda.is_available():
            raise RuntimeError("Paired PROTEUS screen requires a CUDA worker")
        torch.set_num_threads(2)
    predictions, audits = {}, {}
    for i, key in enumerate(SINGLES):
        val = np.isin(rows, groups[key])
        fit = ~val
        if set(people[fit]) & set(people[val]):
            raise ValueError("Fitting and validation participants overlap")
        if lane == "paired":
            pred, audit = fit_predict(cov[fit], cov[val], y[fit], people[fit], interface[fit], mode[fit], seed + i)
        else:
            pred, audit = geometry(cov, fit, val, y), {}
        for arm, logits in pred.items():
            predictions.setdefault(arm, np.full((len(rows), 3), np.nan))[val] = logits
        audits[key] = {"fit_people": np.unique(people[fit]).tolist(),
                       "validation_people": np.unique(people[val]).tolist(), **audit}
    scores = {arm: score(logits, y, people, interface) for arm, logits in predictions.items()}
    report = {"dataset": "PROTEUS", "lane": lane, "seed": seed, "excluded": EVAL, "source_sessions": ["1"],
              "rows": len(rows), "source_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
              "folds": audits, "scores": scores}
    save(output, report)
    print(json.dumps({arm: {scope: d["ba"] for scope, d in res.items()} for arm, res in scores.items()}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("lane", choices=("paired", "geometry"))
    p.add_argument("seed", type=int, choices=(0, 100, 200))
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.lane, a.seed, a.output)
