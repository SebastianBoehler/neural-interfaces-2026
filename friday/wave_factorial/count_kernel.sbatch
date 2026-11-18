#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 8
#SBATCH -J ni26-count-kernel
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
python="$base/neural-interfaces-2026/.venv/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum --quiet -c source.sha256
if [[ ${1:-} == --smoke ]]; then
    (cd "$base/ni26_s318_coral_fbl_20261117" && "$python" replay/check.py) > recording-replay.tmp
    grep -Fx OK recording-replay.tmp
    mv recording-replay.tmp recording-replay.log
fi
grep -Fx OK recording-replay.log
"$python" count_kernel.py "$@"
