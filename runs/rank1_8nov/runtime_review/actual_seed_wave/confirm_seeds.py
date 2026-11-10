"""Check seed confirmation for frozen primary switches on matching overlap pairs."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from numpy.lib import format as npformat

from actual_gate import MIN_GAIN, decisions
from plan import BUILDS, EVAL, LATE, SINGLES, metadata, partition, protocol, source_rows
from score import metrics, slice_stats

OFFSETS = (0, 100, 200)
HASH = re.compile(r"[0-9a-f]{64}")


def artifact(path, hashes, name):
    payload = (path / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != hashes.get(name):
        raise ValueError(f"SHA-256 mismatch or missing manifest entry for {path / name}")
    return payload


def manifest(path):
    hashes = {}
    for line in (path / "outputs.sha256").read_text().splitlines():
        value, name = line.split()
        if HASH.fullmatch(value) is None or name in hashes:
            raise ValueError(f"Malformed or repeated checksum in {path / 'outputs.sha256'}")
        hashes[name] = value
    return hashes


def verify_receipts(path, hashes, offset, meta):
    proof = json.loads(artifact(path, hashes, "source_provenance.json"))
    if set(proof) != set(BUILDS):
        raise ValueError(f"Unexpected build set for seed {offset}")
    geometry = {}
    for key in BUILDS:
        receipt = proof[key]["fit"]
        scripts = proof[key]["torchscript_sha256"]
        rows = source_rows(meta, key)
        if (receipt["key"] != key or receipt.get("seed_offset", 0) != offset
                or tuple(receipt["excluded"]) != EVAL or receipt["source_rows"] != rows.tolist()
                or receipt["fit_participants"] != np.unique(meta["subject"][rows]).tolist()
                or len(scripts) != 31):
            raise ValueError(f"Invalid source or seed receipt for seed {offset}, build {key}")
        values = (receipt["geometry_sha256"], receipt["weights_sha256"], *scripts.values())
        if any(HASH.fullmatch(value) is None for value in values):
            raise ValueError(f"Invalid fitted artifact checksum for seed {offset}, build {key}")
        geometry[key] = receipt["geometry_sha256"]
    return geometry


def load_trace(path, hashes, name, expected, group=None):
    from io import BytesIO

    with np.load(BytesIO(artifact(path, hashes, name)), allow_pickle=False) as npz:
        keys = ("indices", "final", "votes", "reps")
        trace = {key: npz[key] for key in keys}
        if group is not None:
            np.testing.assert_array_equal(npz["group"], group)
    np.testing.assert_array_equal(trace["indices"], expected)
    if not np.issubdtype(trace["indices"].dtype, np.integer):
        raise ValueError(f"Invalid index dtype in {path / name}")
    for key, shape in (("final", (len(expected), 3)), ("reps", (len(expected), 4, 3))):
        p = trace[key]
        if (p.shape != shape or not np.isfinite(p).all() or (p < 0).any()
                or not np.allclose(p.sum(axis=-1), 1, atol=1e-7, rtol=0)):
            raise ValueError(f"Invalid {key} posterior in {path / name}")
    votes = trace["votes"]
    if (votes.shape != (len(expected), 5) or not np.issubdtype(votes.dtype, np.integer)
            or not np.isin(votes, [0, 1, 2]).all()):
        raise ValueError(f"Invalid family votes in {path / name}")
    return trace


def confirm(traces, gates):
    seeded = [decisions(trace, gate) for trace, gate in zip(traces, gates, strict=True)]
    primary, broad, runner = seeded[0]
    winner, direct = primary["identity"], primary["direct"]
    proposed = broad & (direct == runner)
    assert np.array_equal(proposed, direct != winner)

    def supports(current):
        labels, eligible, other_runner = current
        return eligible & (labels["identity"] == winner) & (other_runner == runner) & (labels["direct"] == runner)

    support = np.stack([supports(current) for current in seeded[1:]], axis=1)
    accepted = proposed & support.any(axis=1)
    selected = np.where(accepted, runner, winner)
    assert np.array_equal(selected != winner, accepted)
    assert np.all((selected == direct) | (selected == winner))
    assert not np.any(accepted & ~proposed)
    return winner, direct, selected, broad, runner, proposed, accepted, support


def read_labels(rows, n_rows):
    """Materialize only requested public labels, including when outer scoring is skipped."""
    with ZipFile(protocol.DATA) as archive, archive.open("y.npy") as labels:
        if npformat.read_magic(labels) != (1, 0):
            raise ValueError("Expected the fixed public label array in NPY 1.0 format")
        shape, fortran_order, dtype = npformat.read_array_header_1_0(labels)
        if shape != (n_rows,) or fortran_order or not np.issubdtype(dtype, np.integer):
            raise ValueError("Unexpected public label array")
        start = labels.tell()
        spans = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
        selected = []
        for span in spans:
            labels.seek(start + int(span[0]) * dtype.itemsize)
            selected.append(np.frombuffer(labels.read(len(span) * dtype.itemsize), dtype=dtype))
    result = np.concatenate(selected)
    if result.shape != rows.shape or not np.isin(result, [0, 1, 2]).all():
        raise ValueError("Invalid labels in requested public slice")
    return result


def changes(before, after, truth):
    switched = before != after
    rescues = switched & (after == truth)
    harms = switched & (before == truth)
    return {"switches": int(switched.sum()), "rescues": int(rescues.sum()), "harms": int(harms.sum()),
            "neither": int((switched & ~rescues & ~harms).sum())}


def source_stats(meta, rows, result):
    truth = read_labels(rows, len(meta["subject"]))
    winner, direct, selected, broad, _, proposed, accepted, support = result
    participants = np.array([name.rsplit("/", 1)[-1] for name in meta["subject"][rows]])

    def summarize(at):
        proposed_at, accepted_at = proposed[at], accepted[at]
        removed = proposed_at & ~accepted_at
        return {
            "eligible": int(broad[at].sum()),
            "primary_switches": changes(winner[at], direct[at], truth[at]),
            "confirmed_switches": changes(winner[at], selected[at], truth[at]),
            "removed_primary_switches": changes(winner[at][removed], direct[at][removed], truth[at][removed]),
            "primary_switches_with_confirmation_count": np.bincount(support[at][proposed_at].sum(1), minlength=3).tolist(),
        }

    return {
        "scope": "Descriptive only. The gates were fitted with these source outcomes on source OOF solver traces.",
        "all": summarize(np.ones(len(rows), dtype=bool)),
        "participants": {name: summarize(participants == name) for name in np.unique(participants)},
    }


def evaluation_stats(meta, target, later, primary_final, result):
    truth = read_labels(target, len(meta["subject"]))
    meta = {**meta, "y": np.full(len(meta["subject"]), -1, dtype=truth.dtype)}
    meta["y"][target] = truth
    late_positions = np.searchsorted(target, later)
    np.testing.assert_array_equal(target[late_positions], later)
    winner, direct, selected, broad, _, _, _, _ = result
    scored = {}
    for label, baseline, before in (("seed0_overlap", primary_final, winner), ("seed0_direct", np.eye(3)[direct], direct)):
        slices = {}
        for name, at in (("unseen", np.arange(len(target))), ("later_unseen", late_positions)):
            m = metrics(meta, target[at], baseline[at], np.eye(3)[selected[at]])
            scopes = {
                "all": (slice_stats(m), np.ones(len(at), dtype=bool)),
                "P10": (slice_stats(m, "P10"), np.char.endswith(meta["subject"][target[at]], "/P10")),
                **{interface: (slice_stats({"people": people}), meta["interface"][target[at]] == interface)
                   for interface, people in m["people_by_interface"].items()},
            }
            slices[name] = {
                scope: {**summary, "windows": int(mask.sum()), "eligible": int(broad[at][mask].sum()),
                        **changes(before[at][mask], selected[at][mask], truth[at][mask])}
                for scope, (summary, mask) in scopes.items()
            }
        scored[label] = {"J_delta": float(np.mean([detail["all"]["delta"] for detail in slices.values()])), **slices}
    return scored


def run(paths):
    meta = metadata()
    groups, target, later = partition(meta)
    source = source_rows(meta, "abc")
    group = np.full(len(source), -1, dtype=int)
    for i, key in enumerate(SINGLES):
        group[np.searchsorted(source, groups[key])] = i
    assert (group >= 0).all() and not np.intersect1d(source, target).size

    hashes = [manifest(path) for path in paths]
    geometry = [verify_receipts(path, check, offset, meta)
                for path, check, offset in zip(paths, hashes, OFFSETS, strict=True)]
    if any(d != geometry[0] for d in geometry[1:]):
        raise ValueError("Seed repeats did not use the primary deterministic geometry")
    gates, scores, source_traces = [], [], []
    for path, check in zip(paths, hashes, strict=True):
        gate = json.loads(artifact(path, check, "gate.json"))
        report = json.loads(artifact(path, check, "score.json"))
        if (tuple(report["eval_participants"]) != EVAL or tuple(report["later_unseen_participants"]) != LATE
                or report["source_fit_rows"] != {k: v["fit_rows"] for k, v in gate.items()}
                or report["source_outcomes"] != {k: v["outcomes"] for k, v in gate.items()}):
            raise ValueError(f"Score, frozen gate and split disagree for {path}")
        delta = float(report["arms"]["direct"]["J_delta"])
        if not np.isfinite(delta):
            raise ValueError(f"Nonfinite repeat direct J delta in {path}")
        gates.append(gate)
        scores.append(delta)
        source_traces.append(load_trace(path, check, "source_oof.npz", source, group))
    source_result = confirm(source_traces, gates)
    should_score = all(delta > 0 for delta in scores[1:])
    if should_score:
        outer = [load_trace(path, check, "outer.npz", target) for path, check in zip(paths, hashes, strict=True)]
        outer_result = confirm(outer, gates)
    report = {
        "rule": "Seed 0 direct switch plus at least one independently eligible same ordered pair direct confirmation.",
        "threshold": MIN_GAIN, "seed_direct_J_delta": dict(zip(OFFSETS, scores, strict=True)),
        "no_new_switches": True, "source": source_stats(meta, source, source_result),
    }
    if should_score:
        report.update(status="scored", evaluation=evaluation_stats(meta, target, later, outer[0]["final"], outer_result))
    else:
        report.update(status="skipped", reason="Both repeat direct J deltas must be strictly positive.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("seed100", type=Path)
    parser.add_argument("seed200", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run((args.primary, args.seed100, args.seed200))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "seed_direct_J_delta": report["seed_direct_J_delta"]}), flush=True)


if __name__ == "__main__":
    main()
