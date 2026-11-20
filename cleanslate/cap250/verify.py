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

sys.path[:0] = [str(ROOT / "cleanslate/harness"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import data as harness
from stream import load_solver, stream_sessions

SEALED = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"


def labels(pkg, data, orders):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_solver(pkg, data["ch_names"], device)
    x = torch.from_numpy(data["X"])
    streams = tuple(dict.fromkeys(data["stream"]))
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
    return results


def verify(package, sealed, data_path, out, orders):
    if file_digest(sealed) != SEALED:
        raise ValueError("sealed reference ZIP SHA-256 changed")
    data = harness.load(data_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp, ZipFile(sealed) as archive:
        reference = Path(tmp)
        archive.extractall(reference)
        sealed_predictions = labels(reference, data, orders)
        rebuilt_predictions = labels(package, data, orders)
    result = {}
    changed_rows = {}
    for seed in orders:
        changed = {}
        for visit, (rows, stock, _) in sealed_predictions[seed].items():
            recovered_rows, candidate, _ = rebuilt_predictions[seed][visit]
            if not np.array_equal(rows, recovered_rows):
                raise ValueError(f"delivery mismatch for {seed} / {visit}")
            changed[str(visit)] = int((stock != candidate).sum())
            changed_rows[f"{seed}|{visit}"] = rows[stock != candidate]
        result["recording" if seed is None else seed] = dict(total=sum(changed.values()), visits=changed)
    out.write_text(json.dumps(dict(sealed=SEALED, rebuilt_weights=file_digest(package / "weights.npz"),
                                   orders=result), indent=2) + "\n")
    np.savez(out.with_suffix(".npz"), **changed_rows)
    print(json.dumps(result), flush=True)
    if any(row["total"] for row in result.values()):
        raise SystemExit("fresh package changed public decisions relative to sealed stock")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=default_data())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shuffle", type=int, action="append", default=[])
    args = parser.parse_args()
    verify(args.pkg, args.sealed, args.data, args.out, [None, *args.shuffle])
