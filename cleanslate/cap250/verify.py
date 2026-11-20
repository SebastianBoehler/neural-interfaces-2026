"""Compare every public decision from a rebuilt package and the frozen stock ZIP."""

import argparse
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

from source import ROOT, default_data, file_digest

sys.path[:0] = [str(ROOT / "cleanslate/harness"), str(ROOT / "2026-competition/tracks/bci_decoding"), str(ROOT / "2026-competition")]
import data as harness
from stream import load_solver, stream_sessions

SEALED = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"


def labels(pkg, data, orders, person):
    if not torch.cuda.is_available():
        raise RuntimeError("public cap250 parity requires a CUDA Slurm allocation")
    device = "cuda:0"
    model = load_solver(pkg, data["ch_names"], device)
    x = torch.from_numpy(data["X"])
    allowed = data["stream"] if person is None else data["stream"][data["person"] == person]
    streams = tuple(dict.fromkeys(allowed))
    if not streams:
        raise ValueError(f"no public data for {person}")
    results = {}
    for seed in orders:
        plan = (harness.plans(data, seed, streams) if seed is not None else
                {session: np.flatnonzero(data["stream"] == session) for session in streams})
        errors = io.StringIO()
        with redirect_stderr(errors):
            trial = stream_sessions(model, x, plan, device)
        error_counts = re.findall(r"\[v6\].* errors (\d+)", errors.getvalue())
        if len(error_counts) != len(streams) or any(int(n) != 0 for n in error_counts):
            raise RuntimeError(f"missing v6 status or error on order {seed}: {errors.getvalue()}")
        results[seed] = trial
        elapsed = sum(row[2] for row in trial.values())
        print(f"{pkg}: order {seed if seed is not None else 'recording'}, {len(trial)} visits, {elapsed:.1f}s",
              flush=True)
    return results


def verify(package, sealed, data_path, out, orders, person=None, report_only=False):
    if file_digest(sealed) != SEALED:
        raise ValueError("sealed reference ZIP SHA-256 changed")
    data = harness.load(data_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp, ZipFile(sealed) as archive:
        reference = Path(tmp)
        archive.extractall(reference)
        sealed_predictions = labels(reference, data, orders, person)
        rebuilt_predictions = labels(package, data, orders, person)
    result = {}
    changed_rows = {}
    for seed in orders:
        changed, n_windows = {}, 0
        old = np.full(len(data["X"]), -1, np.int8)
        new = old.copy()
        for visit, (rows, stock, _) in sealed_predictions[seed].items():
            recovered_rows, candidate, _ = rebuilt_predictions[seed][visit]
            if not np.array_equal(rows, recovered_rows):
                raise ValueError(f"delivery mismatch for {seed} / {visit}")
            changed[str(visit)] = int((stock != candidate).sum())
            changed_rows[f"{seed}|{visit}"] = rows[stock != candidate]
            old[rows], new[rows] = stock, candidate
            n_windows += len(rows)
        mask = old >= 0
        opts = (mask, seed, "all")
        actual = [harness.cell_table(data, predicted, *opts) for predicted in (old, new)]
        result["recording" if seed is None else seed] = dict(total=sum(changed.values()), visits=changed,
            windows=n_windows, cells=len(actual[0]), sealed_ba_sum=sum(row["ba"] for row in actual[0]),
            rebuilt_ba_sum=sum(row["ba"] for row in actual[1]))
    out.write_text(json.dumps(dict(sealed=SEALED, rebuilt_weights=file_digest(package / "weights.npz"),
                                   person=person, orders=result), indent=2) + "\n")
    np.savez(out.with_suffix(".npz"), **changed_rows)
    print(json.dumps(result), flush=True)
    if not report_only and any(row["total"] for row in result.values()):
        raise SystemExit("fresh package changed public decisions relative to sealed stock")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shuffle", type=int, action="append", default=[])
    parser.add_argument("--person")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    verify(args.pkg, args.sealed, args.data, args.out, [None, *args.shuffle], args.person, args.report_only)
