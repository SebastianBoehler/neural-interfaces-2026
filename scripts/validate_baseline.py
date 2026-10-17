"""Package and replay the unchanged official baseline on real PROTEUS data."""

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from benchopt.results import read_results

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "2026-competition"
RUN = ROOT / "runs" / "baseline"
EXPORT = REPO / "tracks" / "bci_decoding" / "outputs" / "Torch-Linear"
METRICS = (
    "balanced_accuracy_cell_mean",
    "balanced_accuracy_session_mean",
    "balanced_accuracy",
    "accuracy",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(args, log):
    with log.open("w") as stream:
        subprocess.run([sys.executable, *map(str, args)], cwd=REPO,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    expected_files = {"submission.py", "weights.pt"}
    actual_files = {p.name for p in EXPORT.iterdir() if p.is_file()}
    assert actual_files == expected_files, actual_files
    official = REPO / "tracks/bci_decoding/solvers/torch_linear.py"
    assert digest(EXPORT / "submission.py") == digest(official)
    state = torch.load(EXPORT / "weights.pt", weights_only=True,
                       map_location="cpu")
    assert state["linear.weight"].shape == (3, 41)
    assert state["linear.bias"].shape == (3,)
    assert all(torch.isfinite(t).all() for t in state.values())

    archive = RUN / "proteus-torch-linear.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(expected_files):
            zf.write(EXPORT / name, arcname=name)
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == expected_files
        assert zf.testzip() is None
        replay_dir = Path(tempfile.mkdtemp(prefix="submission-", dir=RUN))
        zf.extractall(replay_dir)

    before = {name: digest(replay_dir / name) for name in expected_files}
    for name in expected_files:
        (replay_dir / name).chmod(0o444)
    replay_dir.chmod(0o555)
    execute([
        REPO / "codabench/ingestion_program/ingestion.py",
        "--submission-dir", replay_dir,
        "--benchmark-dir", REPO / "tracks/bci_decoding",
        "--input-data", REPO / "codabench/phases/warmup/bci_decoding",
        "--output-dir", RUN / "ingestion",
    ], RUN / "ingestion.log")
    execute([
        REPO / "codabench/scoring_program/scoring.py",
        "--prediction-dir", RUN / "ingestion",
        "--output-dir", RUN / "scoring",
    ], RUN / "scoring.log")
    assert before == {name: digest(replay_dir / name)
                      for name in expected_files}

    training_files = sorted((REPO / "tracks/bci_decoding/outputs").glob(
        "baseline*.parquet"), key=lambda p: p.stat().st_mtime)
    train = read_results(training_files[-1]).iloc[0]
    replay = read_results(RUN / "ingestion/results.parquet").iloc[0]
    scores = json.loads((RUN / "scoring/scores.json").read_text())
    for metric in METRICS:
        col = "objective_" + metric
        assert np.isfinite(float(replay[col])), col
        assert np.isclose(float(train[col]), float(replay[col]),
                          rtol=0, atol=1e-12), col
    assert scores["balanced_accuracy_cell_mean"] == float(
        replay["objective_balanced_accuracy_cell_mean"])
    for col in ("objective_n_classes", "objective_n_cells"):
        assert int(train[col]) == int(replay[col]), col
    result = {
        "status": "validated locally; not submitted",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "dataset": "PROTEUS nm000290 v1.0.0",
        "training_config": "tracks/bci_decoding/training.yml",
        "device": "cpu",
        "solver_source_unchanged": True,
        "zip_sha256": digest(archive),
        "zip_size_bytes": archive.stat().st_size,
        "replayed_submission_directory": str(replay_dir.relative_to(ROOT)),
        "submission_files_unchanged_after_replay": True,
        "training_matches_inference_replay": True,
        "metrics": {m: float(replay["objective_" + m]) for m in METRICS},
        "n_cells": int(replay["objective_n_cells"]),
        "scores": scores,
        "limitation": "Native CPU replay; Linux A100 worker not exercised.",
    }
    (RUN / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
