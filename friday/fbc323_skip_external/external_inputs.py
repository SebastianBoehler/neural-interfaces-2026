"""Authenticate the existing Scherer raw cohort and its fixed delivery orders."""
import json
from pathlib import Path

import numpy as np

from data import permutation
from replay_f13 import DATA as SOURCE, DATA_SHA as SOURCE_SHA, row_sha, sha


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "neural-interfaces-2026"
CROSS = HERE.parent / "ni26_cross411b_20261118"
DATA = ROOT / "data/scherer2015/windows.npz"
DATA_SHA = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"
ROW_MAP = ROOT / "runs/diag/external_semantics351/scherer_rows.tsv.gz"
ROW_MAP_SHA = "8ea44213aa0add5058312777553f4fa527720381f473f6542cba42eee2dac10d"
EXPORTER_SHA = "bf0495c89e0ce80c911d26a511762ebb6125f25fbd93c3438352e459d58d9ad7"
METRIC_SHA = "390fd6bea0ddc9c53e34fdabb094b1a269ff45b07b4a5542a8135b44d5314a2c"
ACTOR_SHA = "9d552ab6df96d7aaa171cb523b2a90ba9988f2f698e1544c7802bb960d9ea0e0"
PEOPLE = tuple(f"Scherer2015/{code}" for code in "ACDEFGHJL")
VISITS = ("1", "2")
ORDERS = (411, 1411)
KINDS = ("original", "scaled", "dropout")
MODES = ("ordinary", "bounded")
ARMS = tuple(f"s{seed}_{kind}_{mode}" for seed in (0, 1) for kind in KINDS for mode in MODES)
TOL = 2e-5


def ancestry():
    required = {DATA: DATA_SHA, SOURCE: SOURCE_SHA, ROW_MAP: ROW_MAP_SHA,
                ROOT / "scripts/export_external.py": EXPORTER_SHA,
                ROOT / "experiments/protocol.py": METRIC_SHA, HERE / "actor.sha256": ACTOR_SHA}
    if any(sha(path) != digest for path, digest in required.items()):
        raise ValueError("source, external tensor, row mapping or evaluator hash differs")


def cohort(waveforms):
    with np.load(DATA, allow_pickle=False) as z:
        subject, visit, interface, channels = (z[k] for k in ("subject", "session", "interface", "ch_names"))
        x = z["X"] if waveforms else None
    with np.load(SOURCE, allow_pickle=False) as z:
        if len(channels) != 41 or not np.array_equal(channels, z["ch_names"]):
            raise ValueError("Scherer channels differ from the official PROTEUS order")
    stream = np.char.add(np.char.add(subject, "|"), visit)
    expected = {f"{name}|{day}" for name in PEOPLE for day in VISITS}
    if (subject.shape != (2130,) or set(stream) != expected or
            any(len(set(interface[stream == name])) != 1 for name in expected)):
        raise ValueError("Scherer recording layout differs")
    if waveforms and (x.shape != (2130, 41, 480) or x.dtype != np.float32 or not np.isfinite(x).all()):
        raise ValueError("Scherer raw waveform tensor differs")
    return x, subject, visit, interface, stream


def deliveries(stream, person):
    archive = json.loads((CROSS / f"cap_{PEOPLE.index(person)}.json").read_text())
    result = {}
    for order in ORDERS:
        p = permutation(stream, None, "full", order)
        for visit in VISITS:
            key = f"{person}|{visit}|{order}"
            rows = p[stream == f"{person}|{visit}"]
            if rows.tolist() != archive[key]["rows"]:
                raise ValueError(f"delivery differs from the pinned §411b rows for {key}")
            result[(order, visit)] = rows
    if len(archive) != len(result):
        raise ValueError("unexpected unmatched §411b delivery")
    return result


def delivery_receipt(delivery):
    return {f"{order}|{visit}": row_sha(rows) for (order, visit), rows in delivery.items()}
