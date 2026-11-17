#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --job-name=ni26-csp361
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
task_base=/mnt/lustre/work/gehler/gfh527
task_python="$task_base/neural-interfaces-2026/.venv/bin/python"
sha256sum -c source.sha256
if [[ "$1" == smoke ]]; then
    "$task_python" -B run.py --smoke
elif [[ "$1" == full ]]; then
    cd "$task_base/ni26_s318_coral_fbl_20261117"
    "$task_python" replay/check.py > "$SLURM_SUBMIT_DIR/replay.log"
    grep -Fx OK "$SLURM_SUBMIT_DIR/replay.log"
    cd "$SLURM_SUBMIT_DIR"
    "$task_python" -B run.py
else
    exit 2
fi
sha256sum -c source.sha256
