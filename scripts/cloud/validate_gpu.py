"""Real PROTEUS CUDA training and read-only packaged replay, without uploads."""

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml
from benchopt.results import read_results

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "2026-competition"
TRACK = REPO / "tracks/bci_decoding"
RUN = ROOT / "runs/cloud" / os.environ["SLURM_JOB_ID"]
METRICS = ("balanced_accuracy_cell_mean", "balanced_accuracy_session_mean",
           "balanced_accuracy", "accuracy")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(args, output, device="cuda"):
    started = time.monotonic()
    with output.open("w") as stream:
        subprocess.run(args, cwd=REPO, env={**os.environ, "COMPET_DEVICE": device},
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return time.monotonic() - started


def replay(folder, submission, device):
    target = folder / device
    target.mkdir()
    duration = execute([
        sys.executable, str(REPO / "codabench/ingestion_program/ingestion.py"),
        "--submission-dir", str(submission), "--benchmark-dir", str(TRACK),
        "--input-data", str(REPO / "codabench/phases/warmup/bci_decoding"),
        "--output-dir", str(target / "ingestion"),
    ], target / "ingestion.log", device)
    execute([
        sys.executable, str(REPO / "codabench/scoring_program/scoring.py"),
        "--prediction-dir", str(target / "ingestion"),
        "--output-dir", str(target / "scoring"),
    ], target / "scoring.log", device)
    result = read_results(target / "ingestion/results.parquet").iloc[0]
    assert duration < 3600, "Public replay exceeded the inference budget"
    return result, duration


def validate_solver(name, source):
    folder = RUN / name
    folder.mkdir()
    config = yaml.safe_load((TRACK / "training.yml").read_text())
    config["solver"] = [name]
    config_path = folder / "training.yml"
    config_path.write_text(yaml.safe_dump(config))
    output = f"cloud-{os.environ['SLURM_JOB_ID']}-{name}"
    train_seconds = execute([
        "benchopt", "run", "tracks/bci_decoding", "--config", str(config_path),
        "-r", "1", "--no-cache", "--no-plot", "--no-html", "--no-display",
        "--output", output,
    ], folder / "training.log")
    trained = read_results(TRACK / "outputs" / f"{output}.parquet").iloc[0]
    exported = TRACK / "outputs" / name
    assert digest(exported / "submission.py") == digest(TRACK / "solvers" / source)
    assert {p.name for p in exported.iterdir() if p.is_file()} == {
        "submission.py", "weights.pt"}
    state = torch.load(exported / "weights.pt", weights_only=True, map_location="cpu")
    assert all(torch.isfinite(value).all() for value in state.values())
    archive = folder / "submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name_ in ("submission.py", "weights.pt"):
            zf.write(exported / name_, arcname=name_)
    submission = folder / "submission"
    with zipfile.ZipFile(archive) as zf:
        assert zf.testzip() is None
        zf.extractall(submission)
    before = {p.name: digest(p) for p in submission.iterdir()}
    for path in submission.iterdir():
        path.chmod(0o444)
    submission.chmod(0o555)
    gpu, gpu_seconds = replay(folder, submission, "cuda")
    cpu, cpu_seconds = replay(folder, submission, "cpu")
    assert before == {p.name: digest(p) for p in submission.iterdir()}
    for metric in METRICS:
        key = "objective_" + metric
        assert np.isfinite(float(gpu[key]))
        assert np.isclose(float(trained[key]), float(gpu[key]), rtol=0, atol=1e-12)
    comparison = {metric: float(cpu["objective_" + metric]) -
                  float(gpu["objective_" + metric]) for metric in METRICS}
    record = {
        "solver": name, "status": "GPU training and packaged replay validated",
        "seed": 42, "training_seconds": train_seconds,
        "gpu_replay_seconds": gpu_seconds, "cpu_replay_seconds": cpu_seconds,
        "gpu_metrics": {m: float(gpu["objective_" + m]) for m in METRICS},
        "cpu_minus_gpu_metrics": comparison,
        "cpu_gpu_metrics_identical": all(abs(v) < 1e-12 for v in comparison.values()),
        "zip_sha256": digest(archive), "zip_bytes": archive.stat().st_size,
        "submission_unchanged": True,
        "limitation": "Native Linux, public test; not official Docker or sealed evaluation.",
    }
    (folder / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return record


def main():
    RUN.mkdir(parents=True)
    assert torch.cuda.is_available(), "CUDA must work; CPU execution is not accepted"
    assert torch.cuda.device_count() == 1, "Request exactly one visible GPU"
    sample = torch.randn(64, 64, device="cuda", requires_grad=True)
    sample.square().mean().backward()
    torch.cuda.synchronize()
    hardware = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "official_source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    }
    (RUN / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    with (RUN / "nvidia-smi.txt").open("w") as stream:
        subprocess.run(["nvidia-smi"], stdout=stream, check=True)
    records = [validate_solver("Torch-Linear", "torch_linear.py"),
               validate_solver("EEGNet", "eegnet.py")]
    (RUN / "summary.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
