#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-selector363
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
task_base=/mnt/lustre/work/gehler/gfh527
task_python="$task_base/neural-interfaces-2026/.venv/bin/python"
sha256sum -c source.sha256
cd "$task_base/ni26_s318_coral_fbl_20261117"
"$task_python" replay/check.py > "$SLURM_SUBMIT_DIR/replay.log"
grep -Fx OK "$SLURM_SUBMIT_DIR/replay.log"
cd "$SLURM_SUBMIT_DIR"
"$task_python" -B pilot.py
sha256sum -c source.sha256
