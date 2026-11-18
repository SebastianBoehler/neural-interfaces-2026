#!/bin/bash
# Transfer available F12 weights and release only authenticated paired held jobs.
set -euo pipefail
directory=$(cd "$(dirname "$0")" && pwd)
bash "$directory/relay.sh" band
ssh -o BatchMode=yes mlcloud-galvani 'bash -s' <<'REMOTE'
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
cd "$base/ni26_friday_wave_band_20261118"
sha256sum --quiet -c actor.sha256
sha256sum --quiet -c score.sha256
"$base/neural-interfaces-2026/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess

actor = hashlib.sha256(Path("actor.sha256").read_bytes()).hexdigest()
if actor != "1d919642f3906b7e6d6bbd82c9273add5c96cc8161b786818cd315641d4a80a9":
    raise ValueError("unexpected F12 actor manifest")
queue = subprocess.check_output(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%r"], text=True)
held = {job for line in queue.splitlines() for job, reason in (line.split("|", 1),)
        if reason == "JobHeldUser"}
for line in Path("jobs.score.tsv").read_text().splitlines():
    mode, seed, job = line.split()
    if mode not in ("smoke", "score") or job not in held:
        continue
    prefix = "smoke_" if mode == "smoke" else ""
    people = ("P01",) if mode == "smoke" else tuple(f"P{i:02d}" for i in range(1, 11))
    pairs = [(kind, person, Path("checkpoints") / f"{prefix}{kind}_{seed}_{person}.pt")
             for person in people for kind in ("raw_band_spectral", "raw_band_random")]
    if not all(path.is_file() and path.with_suffix(".json").is_file() for _, _, path in pairs):
        continue
    for kind, person, path in pairs:
        receipt = json.loads(path.with_suffix(".json").read_text())
        expected = dict(actor=actor, held=person, seed=int(seed), kind=kind,
                        steps=4 if mode == "smoke" else 600,
                        checkpoint=hashlib.sha256(path.read_bytes()).hexdigest())
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f"F12 transfer changed: {path}")
    subprocess.run(["scontrol", "release", job], check=True)
    print(f"F12 released {job} {mode} seed {seed} with {len(pairs)} verified checkpoint pairs")
PY
sacct -X -n -j "$(cut -f3 jobs.score.tsv | paste -sd, -)" -o JobID,State,Elapsed,ExitCode
REMOTE
