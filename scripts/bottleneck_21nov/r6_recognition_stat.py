"""Reconstruct the signed IdentityGate statistic on the delivered EEG prefix."""
import ast
from hashlib import file_digest
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from tapes import pinned


BASE_SHA = "833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d"
ROSTER = "ba_six"


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def source_gate(archive):
    if digest(archive) != BASE_SHA:
        raise ValueError("signed em_half ZIP differs")
    with ZipFile(archive) as source:
        tree = ast.parse(source.read("submission.py"))
    names = {"IdentityGate", "channel_map", "ramp"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in names
             or isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "CH_NAMES"
                                                     for target in node.targets)]
    if len(nodes) != 4:
        raise ValueError("missing or duplicate signed identity definitions")
    scope = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(archive), "exec"), scope)
    return scope["IdentityGate"], scope["channel_map"]


def fold_gate(fixtures, work, gate_class, case):
    fold = f"{case.cohort}_{case.person}"
    _, _, receipt = pinned(work / "leo", fold, ROSTER, case.order, f"{case.order}_{case.visit}")
    path = fixtures / fold
    local = json.loads((path / "receipt.json").read_text())[ROSTER]
    asset = path / ROSTER / "weights.npz"
    if local != json.loads(receipt) or digest(asset) != local["assets"]["weights.npz"]:
        raise ValueError(f"unexpected identity gallery for {fold}")
    with np.load(asset, allow_pickle=False) as z:
        people = np.asarray([int(name[1:]) for name in z["route_names"]])
        gate = gate_class(z)
        contrast_start = int(z["route_t_start"]) + 1
    present = np.count_nonzero(people == int(case.person[1:]))
    if present != int(case.cohort == "ret") or len(people) != len(gate.G):
        raise ValueError(f"incorrect enrolment for {fold}")
    return gate, people, contrast_start


def fingerprints(data, gate, channel_map):
    with np.load(data, allow_pickle=False) as z:
        signals = z["X"]
        perm = channel_map(z["ch_names"])
    if perm is not None:
        aligned = np.zeros_like(signals)
        aligned[:, perm >= 0] = signals[:, perm[perm >= 0]]
        signals = aligned
    np.nan_to_num(signals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.stack([gate.feat(row) for row in signals])


def check_native(case, values, people, gate, contrast_start, live_gate):
    ages = np.arange(1, len(values) + 1)
    score_cross = np.flatnonzero((ages > gate.t0) & (values[:, 1] > gate.tau)) + 1
    first = int(score_cross[0]) if len(score_cross) else len(values) + 1
    actual = int(case.scores["rec"][0])
    expected_identity = int(case.scores["identity"][0])
    if actual < 0:
        if first <= len(values):
            raise ValueError(f"{case.key}: reconstructed identity crosses on a native-unlatched stream")
    elif first < actual or (actual < contrast_start and
                            (actual != first or int(values[first - 1, 0]) != expected_identity)):
        raise ValueError(f"{case.key}: reconstructed native latch differs")
    if live_gate is not None:
        identity = live_gate[:, 1]
        active = identity[:, 0] >= 0
        active_ages = ages[:len(identity)]
        np.testing.assert_array_equal(active, active_ages > gate.t0)
        v, witness = values[:len(identity)][active], identity[active]
        np.testing.assert_array_equal(people[witness[:, 0].astype(int)], v[:, 0])
        np.testing.assert_allclose(witness[:, 1], v[:, 1], rtol=0, atol=1e-10)
        np.testing.assert_allclose(witness[:, 2], np.clip((v[:, 1] - gate.tau) / gate.width, 0, 1),
                                   rtol=0, atol=1e-10)
    return {"identity_first": first if first <= len(values) else None, "contrast_start": contrast_start}


def reconstruct(data, work, fixtures, archive, cases, witnesses):
    gate_class, channel_map = source_gate(archive)
    folds = {}
    for case in cases:
        key = case.cohort, case.person
        if key not in folds:
            folds[key] = fold_gate(fixtures, work, gate_class, case)
    features = fingerprints(data, next(iter(folds.values()))[0], channel_map)
    result, native = {}, {}
    for case in cases:
        gate, people, contrast_start = folds[(case.cohort, case.person)]
        gate.reset_state()
        values = np.empty((len(case.rows), 2))
        for i, row in enumerate(case.rows):
            gate.sum = gate.sum + features[row]
            gate.n += 1
            s = gate.scores()
            j = int(s.argmax())
            values[i] = people[j], s[j]
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite cumulative identity statistic: {case.key}")
        native[case.key] = check_native(case, values, people, gate, contrast_start, witnesses.get(case.key))
        result[case.key] = values
    return result, native


def stable_minima(trace):
    """Three consecutive windows for the same winning identity."""
    ids, scores = trace.T
    matches = (ids[:-2] == ids[1:-1]) & (ids[1:-1] == ids[2:])
    minima = np.minimum(np.minimum(scores[:-2], scores[1:-1]), scores[2:])
    return np.where(matches, minima, -np.inf)


def thresholds(traces, cases):
    minima = {case.key: stable_minima(traces[case.key]) for case in cases}
    lengths = [len(bank) for bank in minima.values()]
    maxima = np.full(max(lengths), -np.inf)
    fit_maxima = maxima.copy()
    for key, bank in minima.items():
        if key[0] == "new":
            np.maximum(maxima[:len(bank)], bank, out=maxima[:len(bank)])
            if key[1] < 71808:
                np.maximum(fit_maxima[:len(bank)], bank, out=fit_maxima[:len(bank)])
    if not np.isfinite(maxima.max()) or not np.isfinite(fit_maxima.max()):
        raise ValueError("missing stable impostor stretches")
    # Keep an endpoint closed when the reference block had no stable impostor triple there.
    # All endpoints are evaluated for the global rule because its threshold is shared.
    endpoints = np.where(np.isfinite(maxima), maxima, np.inf)
    fit_endpoints = np.where(np.isfinite(fit_maxima), fit_maxima, np.inf)
    return minima, {"global": (maxima.max(), fit_maxima.max()),
                    "per_delivery": (endpoints, fit_endpoints)}


def crossings(traces, minima, threshold):
    hits = {}
    for key, votes in minima.items():
        limit = threshold[:len(votes)] if np.ndim(threshold) else threshold
        found = np.flatnonzero(votes > limit)
        if len(found):
            i = int(found[0]) + 2
            hits[key] = (i + 1, int(traces[key][i, 0]))
    return hits
