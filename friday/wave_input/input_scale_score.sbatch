#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 10
#SBATCH -J ni26-input-scale
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
python="$base/neural-interfaces-2026/.venv/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum --quiet -c score.sha256
if [[ $1 == preflight ]]; then
    (cd "$base/ni26_s318_coral_fbl_20261117" && "$python" replay/check.py) > recording-replay.tmp
    grep -Fx OK recording-replay.tmp
    mv recording-replay.tmp recording-replay.log
else
    sha256sum --quiet -c actor.sha256
fi
grep -Fx OK recording-replay.log
"$python" input_scale_score.py "$@"
