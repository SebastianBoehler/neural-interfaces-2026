"""Compare a frozen candidate with the baseline on caller-supplied rows."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from screen_metrics import compare as person_accuracy


def _labels(values, n, name):
    values = np.asarray(values)
    if values.shape == (n, 3):
        if (not np.isfinite(values).all() or np.any(values < 0)
                or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-6)):
            raise ValueError(f"{name} must contain finite normalized probabilities")
        return values.argmax(axis=1)
    if (values.shape != (n,) or values.dtype.kind not in "iu"
            or np.any((values < 0) | (values >= 3))):
        raise ValueError(f"{name} must contain aligned three-class labels or posteriors")
    return values


def _summary(people, y, old, new):
    score = float(np.mean([x["score"] for x in people]))
    delta = float(np.mean([x["delta"] for x in people]))
    changed = old != new
    return {"score": score, "baseline": score - delta, "delta": delta,
            "windows": len(y), "switches": int(changed.sum()),
            "rescues": int((changed & (new == y)).sum()),
            "regressions": int((changed & (old == y)).sum())}


def _section(meta, rows, old, new):
    y = meta["y"][rows]
    people = meta["subject"][rows]
    interfaces = meta["interface"][rows]
    eye = np.eye(3)
    results = person_accuracy(meta, rows, eye[old], eye[new])
    names = {person: person.rsplit("/", 1)[-1] for person in np.unique(people)}
    if len(names) != len(results):
        raise ValueError("Participant suffixes must be unique within the score")
    report = {"all": _summary(results.values(), y, old, new),
              "participants": {}, "interfaces": {}}
    for person, name in names.items():
        at = people == person
        report["participants"][name] = _summary([results[name]], y[at], old[at], new[at])
    for interface in np.unique(interfaces):
        at = interfaces == interface
        subset = person_accuracy(meta, rows[at], eye[old[at]], eye[new[at]])
        report["interfaces"][str(interface)] = _summary(subset.values(), y[at], old[at], new[at])
    if "P10" in report["participants"]:
        report["P10"] = report["participants"]["P10"]
    return report


def compare(meta, rows, baseline, candidate, slices=None):
    """Score global row indices and optional named, local boolean cohort masks.

    Accuracy and deltas are in percentage points. J_delta gives each named
    slice equal weight and defaults to the overall delta when none are supplied.
    """
    meta = {key: np.asarray(meta[key]) for key in ("y", "subject", "stream", "interface")}
    n = len(meta["y"])
    if any(value.shape != (n,) for value in meta.values()):
        raise ValueError("Score metadata must have one aligned entry per dataset row")
    rows = np.asarray(rows)
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows)
            or np.any(rows < 0) or np.any(rows >= n) or len(np.unique(rows)) != len(rows)):
        raise ValueError("Expected distinct nonempty global row indices")
    _labels(meta["y"][rows], len(rows), "Truth")
    old = _labels(baseline, len(rows), "Baseline")
    new = _labels(candidate, len(rows), "Candidate")
    overall = _section(meta, rows, old, new)
    scores = {}
    for name, at in (slices or {}).items():
        at = np.asarray(at)
        if at.shape != rows.shape or at.dtype != np.bool_ or not at.any():
            raise ValueError(f"Slice {name!r} must select rows with an aligned boolean mask")
        scores[name] = _section(meta, rows[at], old[at], new[at])
    j = float(np.mean([x["all"]["delta"] for x in scores.values()])) if scores else overall["all"]["delta"]
    return {"overall": overall, "slices": scores, "J_delta": j}
