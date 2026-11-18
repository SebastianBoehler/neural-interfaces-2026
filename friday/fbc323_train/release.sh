#!/bin/bash
# Transfer authentic completed S1C checkpoints and release only the matching held jobs.
set -euo pipefail
directory=$(cd "$(dirname "$0")" && pwd)
bash "$directory/../wave/relay.sh" s1c_train
ssh -o BatchMode=yes -o ConnectTimeout=6 mlcloud-galvani 'bash -s' <<'REMOTE'
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_friday_s1c_training_20261118
sha256sum --quiet -c score.sha256
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


smoke, matrix, _ = Path("jobs.held.txt").read_text().split()
pending = subprocess.check_output(["squeue", "-r", "-h", "-j", f"{smoke},{matrix}", "-o", "%i|%r"], text=True)
for row in pending.splitlines():
    job, reason = row.split("|", 1)
    if reason != "JobHeldUser":
        continue
    if job == smoke:
        tag, seed, held, epochs = "smoke_", 0, "P01", 4
    else:
        task = int(job.rsplit("_", 1)[1])
        tag, seed, held, epochs = "", task // 10, f"P{task % 10 + 1:02d}", 40
    kinds = ("original", "causal", "bounded")
    checkpoints = [Path("checkpoints") / f"{tag}{kind}_{seed}_{held}.pt" for kind in kinds]
    if not all(p.is_file() and p.with_suffix(".json").is_file() for p in checkpoints):
        continue
    receipts = []
    for kind, cp in zip(kinds, checkpoints, strict=True):
        meta = json.loads(cp.with_suffix(".json").read_text())
        expected = dict(kind=kind, seed=seed, held=held, epochs=epochs, actor=sha("actor.sha256"),
                        runtime=sha("runtime.lock"), checkpoint=sha(cp), held_labels_opened=False)
        if any(meta[key] != value for key, value in expected.items()):
            raise ValueError(f"transferred source checkpoint did not authenticate {cp}")
        receipts.append(meta)
    for key in ("initial", "source_rows", "batches", "steps", "scale", "parameters"):
        if len({m[key] for m in receipts}) != 1:
            raise ValueError(f"source fit controls differ for {held} seed {seed}: {key}")
    subprocess.run(["scontrol", "release", job], check=True)
    print(f"Released {job} for {held} seed {seed}")
PY
REMOTE
