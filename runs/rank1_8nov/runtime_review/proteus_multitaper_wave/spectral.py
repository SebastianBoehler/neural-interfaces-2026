"""Matched spectral decoders on source folds with the frozen overlap baseline."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import dpss, hann
from scipy.special import softmax

REVIEW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW / "proteus_mechanism_lanes"))
from lane import (EVAL, FIXED_WEIGHT, OFFSETS, SINGLES, blend, bootstrap, compare,
                  covariances, diagonal, fit_head, load_trace, manifest, metadata,
                  partition, power_predictions, read_eeg, read_labels, save,
                  source_rows, summarize, verify_receipts)
from run import protocol
from screen import BANDS

SEEDS = (0, 100, 200, 300)
PAIRS = (("C1", "C2"), ("CP1", "CP2"), ("P3", "P4"), ("PO7", "PO8"),
         ("P5", "P6"), ("TP7", "TP8"), ("CP5", "CP6"), ("CP3", "CP4"),
         ("C3", "C4"), ("C5", "C6"), ("FC3", "FC4"), ("F3", "F4"))
SCOPES = ("overall", "BrainHero", "Graz")
CONTRASTS = {"multitaper_full": ("hann_full",), "multitaper_power": ("hann_power",),
             "multitaper_lag": ("multitaper_real", "multitaper_power")}


def spectral(x, tapers, left, right):
    """Return relative band power and signed real/imaginary band coherency."""
    n, c, t = x.shape
    tapers = np.asarray(tapers / np.linalg.norm(tapers, axis=-1, keepdims=True),
                        dtype=np.float32)
    f = rfftfreq(t, 1 / 120)
    bands = [(f >= lo) & (f < hi) for lo, hi in BANDS]
    power = np.empty((n, len(BANDS), c), dtype=np.float32)
    coherence = np.empty((n, len(BANDS), len(left)), dtype=np.complex64)
    for start in range(0, n, 24):
        idx = slice(start, start + 24)
        v = x[idx].astype(np.float32)
        v -= v.mean(axis=1, keepdims=True)
        z = rfft(v[:, None, :, :] * tapers[None, :, None, :], axis=-1)
        for b, mask in enumerate(bands):
            s = z[..., mask]
            ps = (s.real**2 + s.imag**2).mean(axis=(1, 3))
            cross = (s[:, :, left] * s[:, :, right].conj()).mean(axis=(1, 3))
            denominator = np.sqrt(ps[:, left]) * np.sqrt(ps[:, right])
            coherence[idx, b] = cross / denominator.clip(1e-20)
            power[idx, b] = np.log((ps / ps.mean(axis=-1, keepdims=True).clip(1e-20)).clip(1e-6))
    if not np.isfinite(power).all() or not np.isfinite(coherence).all():
        raise ValueError("Nonfinite spectral feature")
    return power, coherence


def self_check():
    t = np.arange(480) / 120
    a, b = np.sin(2 * np.pi * 10 * t), np.cos(2 * np.pi * 10 * t)
    x = np.array([np.stack((a, b, a, -2 * a - b))])
    for tapers in (hann(480)[None], dpss(480, 2.5, Kmax=4, sym=False, norm=2)):
        p, z = spectral(x, tapers, [0, 0, 1], [1, 2, 0])
        assert p.shape == (1, 5, 4) and z.shape == (1, 5, 3)
        np.testing.assert_allclose(z[0, 1], [-1j, 1, 1j], atol=2e-3)


def feature_views(raw, channels):
    if raw.shape[1:] != (len(channels), 480):
        raise ValueError("Expected official four-second 120 Hz windows")
    left, right = np.array([[channels.index(ch) for ch in pair] for pair in PAIRS]).T
    n = len(raw)
    views = {}
    for name, tapers in (("hann", hann(480)[None]),
                         ("multitaper", dpss(480, 2.5, Kmax=4, sym=False, norm=2))):
        p, z = spectral(raw, tapers, left, right)
        p, r, i = (a.reshape(n, -1) for a in (p, z.real, z.imag))
        views[f"{name}_power"] = p
        views[f"{name}_full"] = np.column_stack((p, r, i))
        if name == "multitaper":
            views[f"{name}_real"] = np.column_stack((p, r))
            views[f"{name}_lag"] = np.column_stack((p, i))
    return views


def run(seed, paths, prior, output):
    self_check()
    meta = metadata()
    groups, protected, _ = partition(meta)
    rows = source_rows(meta, "abc")
    n = len(meta["subject"])
    people = np.array([p.rsplit("/", 1)[-1] for p in meta["subject"][rows]])
    interface = meta["interface"][rows]
    if (len(rows) != 1323 or np.intersect1d(rows, protected).size or set(people) & set(EVAL)
            or set(meta["session"][rows]) != {"1"}):
        raise ValueError("Protected rows entered the source bank")
    with np.load(protocol.DATA, allow_pickle=False) as bank:
        channels = bank["ch_names"].tolist()
    raw, y = read_eeg(rows, n), read_labels(rows, n)
    meta["y"] = np.full(n, -1, dtype=int)
    meta["y"][rows] = y
    views = feature_views(raw, channels)
    cov = covariances(raw)
    diag = diagonal(cov)
    fold_id = np.full(len(rows), -1, dtype=int)
    power = np.full((len(rows), 3), np.nan)
    predictions = {arm: np.full_like(power, np.nan) for arm in views}
    audit = {}
    for i, key in enumerate(SINGLES):
        held = np.isin(rows, groups[key])
        if not held.any() or set(people[held]) & set(people[~held]):
            raise ValueError("Source fit and validation people overlap")
        fold_id[held] = i
        model = fit_head(diag[~held], y[~held])
        power[held] = power_predictions(model, cov[held], source_oof=True)
        draw = bootstrap(np.flatnonzero(~held), y, people, interface, seed, i)
        for arm, view in views.items():
            model = fit_head(view[draw], y[draw])
            predictions[arm][held] = model.decision_function(view[held])
        audit[key] = {"fit_people": np.unique(people[draw]).tolist(),
                      "validation_people": np.unique(people[held]).tolist(),
                      "draw_sha256": hashlib.sha256(draw.tobytes()).hexdigest()}
    if not np.isfinite(power).all() or not all(np.isfinite(z).all() for z in predictions.values()):
        raise ValueError("Incomplete source fold predictions")
    old = json.loads(prior.read_text())["source_blends"]
    scores = {}
    for path, offset in zip(paths, OFFSETS, strict=True):
        check = manifest(path)
        verify_receipts(path, check, offset, meta)
        overlap = load_trace(path, check, "source_oof.npz", rows, fold_id)["final"]
        fixed = blend(overlap, power, FIXED_WEIGHT)
        if compare(meta, rows, overlap, fixed) != old[str(offset)]["diagonal"]:
            raise ValueError("Fixed 10% source baseline parity failed")
        scores[str(offset)] = {}
        for arm, logits in predictions.items():
            aux = softmax(logits, axis=1)
            scores[str(offset)][arm] = {
                "add": summarize(meta, rows, fixed.argmax(1), blend(fixed, aux, FIXED_WEIGHT).argmax(1)),
                "replace": summarize(meta, rows, fixed.argmax(1), blend(overlap, aux, FIXED_WEIGHT).argmax(1)),
            }
    save(output, {"seed": seed, "source_rows": len(rows), "excluded": EVAL,
                  "folds": audit, "scores": scores})
    print(f"Completed source-only spectral seed {seed} with fixed comparator parity", flush=True)


def aggregate(inputs, output):
    trials = [json.loads(path.read_text()) for path in inputs]
    if [r["seed"] for r in trials] != list(SEEDS) or any(
            r["source_rows"] != 1323 or set(r["excluded"]) != set(EVAL)
            or set(r["scores"]) != {str(o) for o in OFFSETS}
            or any(set(f["fit_people"]) & (set(EVAL) | set(f["validation_people"]))
                   for f in r["folds"].values()) for r in trials):
        raise ValueError("Invalid source experiment receipts")
    arms = trials[0]["scores"]["0"]
    by_mode = {}
    for mode in ("add", "replace"):
        means = {}
        for a in arms:
            per_seed = [{k: float(np.mean([s[a][mode][k]["delta"] for s in r["scores"].values()]))
                         for k in SCOPES} for r in trials]
            means[a] = {"by_seed": per_seed, "mean": {
                k: float(np.mean([row[k] for row in per_seed])) for k in SCOPES}}
        gates = {}
        for candidate, controls in CONTRASTS.items():
            row = means[candidate]["mean"]
            gains = {c: row["overall"] - means[c]["mean"]["overall"] for c in controls}
            checks = {"overall_0p8": row["overall"] >= .8, "brainhero_0p5": row["BrainHero"] >= .5,
                      "graz_nonnegative": row["Graz"] >= 0, "matched_controls_0p3": min(gains.values()) >= .3,
                      "three_positive_seeds": sum(s["overall"] > 0 for s in means[candidate]["by_seed"]) >= 3}
            gates[candidate] = {"pass": all(checks.values()), "checks": checks, "control_gains": gains}
        by_mode[mode] = {"arms": means, "source_gate": gates}
    save(output, {"seeds": SEEDS, "excluded": EVAL, "modes": by_mode})
    print(json.dumps(by_mode), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    trial = sub.add_parser("run")
    trial.add_argument("seed", type=int, choices=SEEDS)
    trial.add_argument("archives", nargs=3, type=Path)
    trial.add_argument("prior", type=Path)
    trial.add_argument("output", type=Path)
    summary = sub.add_parser("aggregate")
    summary.add_argument("inputs", nargs=4, type=Path)
    summary.add_argument("output", type=Path)
    a = p.parse_args()
    if a.command == "run":
        run(a.seed, a.archives, a.prior, a.output)
    elif a.command == "aggregate":
        aggregate(a.inputs, a.output)
    else:
        self_check()
