#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-c31e-parent
#SBATCH --output=prepare-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
fit=$base/ni26_friday_c31d_source_fit_v1_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --quiet -c code.sha256
sha256sum --quiet -c scoring.sha256
(cd "$fit" && sha256sum --quiet -c fit.sha256 && sha256sum --quiet -c export.sha256)
python3 - "$fit" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

fit = Path(sys.argv[1])
path = fit/"actors.json"
lock = json.loads(path.read_text())

def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

want = ("hard","matched","nominal","hard_h100")
if (not lock["c31e_source_only"] or lock["held_labels_opened"] or set(lock["actors"]) != set(want)
        or lock["source"] != digest(fit/"fit.sha256") or lock["exporter"] != digest(fit/"export.sha256")
        or path.stat().st_mode & 0o222):
    raise ValueError("CPU actor proof was not published from the authorized research fit")
for arm in want[:-1]:
    if len(lock["actors"][arm]) != 2:
        raise ValueError("expected original independent ordered seed pair")
    for seed,pin in enumerate(lock["actors"][arm]):
        p = fit/"exports"/arm/f"raw_s1c_s{seed}.pt"
        if digest(p) != pin:
            raise ValueError("received actor differs from independently published CPU proof")
for name,sha in lock["bindings"].items():
    if digest(fit/"exports"/name) != sha:
        raise ValueError("received proof hash differs")
PY
ln "$fit/actors.json" .
sha256sum fixture.py pupil_cache.py native_run.py worker.sbatch actors.json > source.sha256
chmod 444 source.sha256
sha256sum --quiet -c source.sha256
sha256sum source.sha256 actors.json
