"""Replay a submission folder through the official ingestion and scoring.

Usage: python scripts/replay_package.py SUBMISSION_DIR|SUBMISSION.zip OUT_DIR [cpu|cuda]
Zips the folder's files at the archive root (like ``zip -j``), extracts the
ZIP into OUT_DIR/extracted, makes it read-only, runs the warm-up phase's
codabench ingestion program (inference-only Benchopt run) and scoring
program, checks the extracted files are unchanged and writes
OUT_DIR/replay.json. Nothing is uploaded.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "2026-competition"
TRACK = REPO / "tracks/bci_decoding"
PHASE = REPO / "codabench/phases/warmup/bci_decoding"


def digests(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.iterdir()) if p.is_file()}


def run(args, log, device):
    t0 = time.monotonic()
    with log.open("w") as f:
        subprocess.run(args, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                       check=True, env={**os.environ, "COMPET_DEVICE": device})
    return round(time.monotonic() - t0, 1)


def main():
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    out.mkdir(parents=True, exist_ok=False)
    archive = out / "submission.zip"
    if src.suffix == ".zip":  # an existing upload ZIP: replay that very file (its bytes, its SHA-256)
        archive.write_bytes(src.read_bytes())
    else:
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(src.iterdir()):
                if p.is_file() and p.suffix in (".py", ".npz", ".pt", ".so"):
                    zf.write(p, arcname=p.name)
    sub = out / "extracted"
    with zipfile.ZipFile(archive) as zf:
        root = sorted(zf.namelist())
        zf.extractall(sub)
    for p in sub.iterdir():
        p.chmod(0o444)
    sub.chmod(0o555)
    before = digests(sub)
    ingest = run([sys.executable, str(REPO / "codabench/ingestion_program/ingestion.py"),
                  "--submission-dir", str(sub), "--benchmark-dir", str(TRACK),
                  "--input-data", str(PHASE), "--output-dir", str(out / "ingestion")],
                 out / "ingestion.log", device)
    run([sys.executable, str(REPO / "codabench/scoring_program/scoring.py"),
         "--prediction-dir", str(out / "ingestion"), "--output-dir", str(out / "scoring")],
        out / "scoring.log", device)
    after = digests(sub)
    scores = json.loads((out / "scoring/scores.json").read_text())
    res = {"submission": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
           "device": device, "zip_root": root, "zip_bytes": archive.stat().st_size,
           "zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
           "files_sha256": before, "files_unchanged": before == after,
           "ingestion_wall_seconds": ingest, "scores": scores}
    (out / "replay.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    assert before == after, "submission files changed during replay"


if __name__ == "__main__":
    main()
