"""Compare the cognitive-pair decoder with the fixed 10% channel-power baseline."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "proteus_channel_power_ba_wave"))
from fit_channel_power_blend import (EVAL, FIXED_WEIGHT, OFFSETS, SINGLES, blend, compare,
                                     covariances, fit_head, load_trace, manifest, metadata, model_sha256,
                                     partition, read_eeg, read_labels, save, source_rows, verify_receipts)
from csp import ARMS, PAIR_WEIGHT, bootstrap_cells, digest, fit, mix, predict


def diagonal(cov):
    return np.log(cov.diagonal(axis1=-2, axis2=-1)).reshape(len(cov), -1)


def power_predictions(model, cov, *, source_oof=False):
    logits = model.decision_function(diagonal(cov))
    if source_oof:
        logits = logits.astype(np.float64)
    return softmax(logits, axis=1)


def cognitive_stats(meta, rows, before, after, slices):
    y = meta["y"][rows]
    out = {}
    for name, scope in slices.items():
        out[name] = {}
        for interface in ("BrainHero", "Graz"):
            at = scope & (meta["interface"][rows] == interface)
            counts = {}
            for truth, wrong in ((1, 2), (2, 1)):
                labeled = at & (y == truth)
                counts[f"{truth}_called_{wrong}"] = {
                    "baseline": int(np.count_nonzero(labeled & (before == wrong))),
                    "candidate": int(np.count_nonzero(labeled & (after == wrong))),
                    "truth_windows": int(labeled.sum()),
                }
            out[name][interface] = counts
    return out


def scores(meta, rows, overlap, power, pair, slices):
    base = blend(overlap, power, FIXED_WEIGHT)
    post = {arm: mix(base, values) for arm, values in pair.items()}
    data = {"fixed_vs_overlap": compare(meta, rows, overlap, base, slices)}
    for arm, candidate in post.items():
        data[f"{arm}_vs_fixed"] = compare(meta, rows, base, candidate, slices)
        scopes = slices or {"source": np.ones(len(rows), dtype=bool)}
        data[f"{arm}_confusion"] = cognitive_stats(meta, rows, base.argmax(1), candidate.argmax(1), scopes)
    data["stable_vs_pooled"] = compare(meta, rows, post["pooled_csp"], post["stable_csp"], slices)
    return data


def run(paths, fixed_fit, fixed_result, fit_path, result_path):
    meta = metadata()
    n = len(meta["subject"])
    groups, target, later = partition(meta)
    source = source_rows(meta, "abc")
    people = np.array([p.rsplit("/", 1)[-1] for p in meta["subject"][source]])
    if (len(source) != 1323 or len(target) != 1332 or len(later) != 762
            or set(people) & set(EVAL) or np.intersect1d(source, target).size
            or set(meta["session"][source]) != {"1"}):
        raise ValueError("PROTEUS split differs from the fixed channel-power replay")
    manifests = [manifest(path) for path in paths]
    provenances = [verify_receipts(path, m, seed, meta)
                   for path, m, seed in zip(paths, manifests, OFFSETS, strict=True)]
    if any(p != provenances[0] for p in provenances[1:]):
        raise ValueError("Overlap geometry differs by seed")

    cov = covariances(read_eeg(source, n))
    y = read_labels(source, n)
    interface = meta["interface"][source]
    meta["y"] = np.full(n, -1, dtype=int)
    meta["y"][source] = y
    folds = {key: np.isin(source, group) for key, group in groups.items()}
    fold_id = np.full(len(source), -1, dtype=int)
    power_oof = np.full((len(source), 3), np.nan)
    for i, key in enumerate(SINGLES):
        held = folds[key]
        if not held.any() or set(people[held]) & set(people[~held]):
            raise ValueError("Source fold overlap")
        fold_id[held] = i
        model = fit_head(diagonal(cov[~held]), y[~held])
        power_oof[held] = power_predictions(model, cov[held], source_oof=True)
    power_head = fit_head(diagonal(cov), y)

    expected_source = json.loads(fixed_fit.read_text())["source_blends"]
    source_scores, source_audits, models = {}, {}, {}
    for path, check, seed in zip(paths, manifests, OFFSETS, strict=True):
        base = load_trace(path, check, "source_oof.npz", source, fold_id)["final"]
        csp_oof = {arm: np.full(len(source), np.nan) for arm in ARMS}
        audit = {}
        for i, key in enumerate(SINGLES):
            held = folds[key]
            indices = np.flatnonzero(~held)
            cells = bootstrap_cells(indices, y, people, interface, seed, i)
            fitted, record = fit(cov, y, cells)
            for arm, value in predict(fitted, cov[held]).items():
                csp_oof[arm][held] = value
            audit[key] = {**record, "fit_people": np.unique(people[indices]).tolist(),
                          "held_people": np.unique(people[held]).tolist()}
        cells = bootstrap_cells(np.arange(len(source)), y, people, interface, seed, len(SINGLES))
        models[seed], audit["full"] = fit(cov, y, cells)
        audit["full"]["model_sha256"] = digest(models[seed])
        source_scores[str(seed)] = scores(meta, source, base, power_oof, csp_oof, None)
        if source_scores[str(seed)]["fixed_vs_overlap"] != expected_source[str(seed)]["diagonal"]:
            raise ValueError("Fixed 10% baseline differs on source predictions")
        source_audits[str(seed)] = audit
    receipt = {"excluded_from_fit_and_tuning": EVAL, "source_people": np.unique(people).tolist(),
               "source_windows": len(source), "power_weight": FIXED_WEIGHT, "pair_weight": PAIR_WEIGHT,
               "power_head_sha256": model_sha256(power_head), "source_audits": source_audits,
               "source_scores": source_scores}
    save(fit_path, receipt)
    print(f"Froze all source heads and fusion weights before opening reserved EEG: {fit_path}", flush=True)

    held_cov = covariances(read_eeg(target, n))
    held_power = power_predictions(power_head, held_cov)
    held_pairs = {seed: predict(model, held_cov) for seed, model in models.items()}
    if (model_sha256(power_head) != receipt["power_head_sha256"]
            or any(digest(m) != source_audits[str(s)]["full"]["model_sha256"] for s, m in models.items())):
        raise ValueError("A frozen model changed during inference")
    slices = {"unseen": np.ones(len(target), dtype=bool), "later_unseen": np.isin(target, later)}
    bases = [load_trace(path, check, "outer.npz", target)["final"]
             for path, check in zip(paths, manifests, strict=True)]
    meta["y"][target] = read_labels(target, n)
    expected_outer = json.loads(fixed_result.read_text())["seed_scores"]
    result = {}
    for seed, base in zip(OFFSETS, bases, strict=True):
        detail = scores(meta, target, base, held_power, held_pairs[seed], slices)
        if detail["fixed_vs_overlap"] != expected_outer[str(seed)]["diagonal"]:
            raise ValueError("Fixed 10% baseline differs on reserved predictions")
        result[str(seed)] = detail
    save(result_path, {"fit_receipt_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                       "excluded_from_fit_and_tuning": EVAL,
                       "windows": {name: int(at.sum()) for name, at in slices.items()},
                       "scores": result,
                       "outer_trace_sha256": {str(seed): m["outer.npz"]
                                              for seed, m in zip(OFFSETS, manifests, strict=True)}})
    print(json.dumps({seed: {key: value["J_delta"] for key, value in detail.items() if "J_delta" in value}
                      for seed, detail in result.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs=3, type=Path)
    for name in ("fixed_fit", "fixed_result", "fit_receipt", "result"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.archives, args.fixed_fit, args.fixed_result, args.fit_receipt, args.result)
