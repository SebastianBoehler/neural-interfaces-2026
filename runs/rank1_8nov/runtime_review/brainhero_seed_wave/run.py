"""Screen a latency-invariant cognitive-pair head with source participants only."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from numpy.lib import format as npformat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "actual_seed_wave"))
from confirm_seeds import load_trace, manifest, read_labels, verify_receipts
from plan import EVAL, LATE, SINGLES, metadata, partition, protocol, source_rows

from burst_head import features, fit_heads, fit_logits, predict_logits
from score_candidate import compare

SEEDS = (0, 100, 200)
ARMS = ("median", "distribution")
ODDS_WEIGHT = 0.5


def read_eeg(rows, n_rows):
    """Read only selected rows from the compressed EEG array."""
    with ZipFile(protocol.DATA) as archive, archive.open("X.npy") as stream:
        if npformat.read_magic(stream) != (1, 0):
            raise ValueError("Expected NPY 1.0 EEG array")
        shape, fortran, dtype = npformat.read_array_header_1_0(stream)
        if shape != (n_rows, 41, 480) or fortran or dtype != np.dtype("float32"):
            raise ValueError("Unexpected EEG array shape, layout or dtype")
        start = stream.tell()
        row_bytes = 41 * 480 * dtype.itemsize
        eeg = np.empty((len(rows), 41, 480), dtype=dtype)
        pos = 0
        for span in np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1):
            stream.seek(start + int(span[0]) * row_bytes)
            block = stream.read(len(span) * row_bytes)
            eeg[pos:pos + len(span)] = np.frombuffer(block, dtype=dtype).reshape(-1, 41, 480)
            pos += len(span)
    return eeg


def tilt_pair(base, logits):
    ranked = np.argsort(-base, axis=1, kind="stable")
    pair = (ranked[:, :2] != 0).all(axis=1)
    baseline_odds = np.log(base[:, 1].clip(1e-12)) - np.log(base[:, 2].clip(1e-12))
    adjusted = baseline_odds + ODDS_WEIGHT * (logits[:, 1] - logits[:, 2])
    labels = np.where(pair & (adjusted != 0), np.where(adjusted > 0, 1, 2), ranked[:, 0])
    assert np.all((labels == ranked[:, 0]) | (pair & (labels == ranked[:, 1])))
    return labels


def source_criteria(reports):
    full = [arms["distribution"] for arms in reports.values()]
    mean = lambda values: float(np.mean(list(values)))
    all_d = [r["overall"]["all"]["delta"] for r in full]
    brain = [r["overall"]["interfaces"]["BrainHero"]["delta"] for r in full]
    graz = [r["overall"]["interfaces"]["Graz"]["delta"] for r in full]
    control = [arms["median"]["overall"]["all"]["delta"] for arms in reports.values()]
    folds = {key: mean(r["slices"][key]["all"]["delta"] for r in full) for key in SINGLES}
    checks = {
        "overall_mean_at_least_0p25": mean(all_d) >= 0.25,
        "brainhero_mean_at_least_0p25": mean(brain) >= 0.25,
        "graz_mean_nonnegative": mean(graz) >= -1e-10,
        "each_seed_overall_nonnegative": min(all_d) >= -1e-10,
        "at_least_two_source_folds_improve": sum(x > 1e-10 for x in folds.values()) >= 2,
        "distribution_beats_median_control": mean(all_d) > mean(control) + 1e-10,
    }
    return {"pass": all(checks.values()), "checks": checks, "mean_fold_delta": folds,
            "mean_delta": {"all": mean(all_d), "BrainHero": mean(brain), "Graz": mean(graz),
                           "median_control_all": mean(control)}}


def score(meta, rows, labels, traces, heads, slices):
    label_vector = np.full(len(meta["subject"]), -1, dtype=int)
    label_vector[rows] = labels
    labeled = {**meta, "y": label_vector}
    return {str(seed): {arm: compare(labeled, rows, trace["final"], tilt_pair(trace["final"], logits), slices)
                        for arm, logits in zip(ARMS, heads, strict=True)}
            for seed, trace in zip(SEEDS, traces, strict=True)}


def save(path, report):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(paths, output):
    meta = metadata()
    groups, target, later = partition(meta)
    source = source_rows(meta, "abc")
    n_rows = len(meta["subject"])
    assert not np.intersect1d(source, target).size
    assert set(meta["session"][source]) == {"1"}
    groups_local = {key: np.isin(source, groups[key]) for key in SINGLES}
    fold = np.full(len(source), -1, dtype=int)
    for i, key in enumerate(SINGLES):
        fold[groups_local[key]] = i
    assert (fold >= 0).all()
    hashes = [manifest(path) for path in paths]
    geometry = [verify_receipts(path, check, seed, meta)
                for path, check, seed in zip(paths, hashes, SEEDS, strict=True)]
    if any(state != geometry[0] for state in geometry[1:]):
        raise ValueError("Compared seeds use different source geometry")
    traces = [load_trace(path, check, "source_oof.npz", source, fold)
              for path, check in zip(paths, hashes, strict=True)]
    source_y = read_labels(source, n_rows)
    stationary, distribution = features(read_eeg(source, n_rows))
    oof = np.full((len(ARMS), len(source), 3), np.nan)
    for held in groups_local.values():
        train, test = np.flatnonzero(~held), np.flatnonzero(held)
        oof[:, test] = fit_logits(stationary, distribution, source_y, train, test)
    assert np.isfinite(oof).all()
    source_result = score(meta, source, source_y, traces, oof, groups_local)
    gate = source_criteria(source_result)
    report = {"status": "source_passed" if gate["pass"] else "source_failed", "excluded": EVAL,
              "later_unseen_people": LATE, "source_sessions": ["1"], "odds_weight": ODDS_WEIGHT,
              "source_window_count": len(source), "source_gate": gate, "source": source_result}
    save(output, report)
    print(json.dumps(gate), flush=True)
    if not gate["pass"]:
        return

    fitted = fit_heads(stationary, distribution, source_y, np.arange(len(source)))
    outer = [load_trace(path, check, "outer.npz", target) for path, check in zip(paths, hashes, strict=True)]
    outer_features = features(read_eeg(target, n_rows))
    logits = predict_logits(fitted, *outer_features)
    outer_labels = read_labels(target, n_rows)
    slices = {"unseen": np.ones(len(target), dtype=bool), "later_unseen": np.isin(target, later)}
    report["heldout"] = score(meta, target, outer_labels, outer, logits, slices)
    report["status"] = "scored"
    save(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("seed100", type=Path)
    parser.add_argument("seed200", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run((args.primary, args.seed100, args.seed200), args.output)
