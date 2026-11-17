#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:12:00
#SBATCH --job-name=ni26-cp365
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
task_home=/mnt/lustre/work/gehler/gfh527
task_python="$task_home/neural-interfaces-2026/.venv/bin/python"
sha256sum -c source.sha256
case ${1:?} in
    smoke)
        "$task_python" -B schedule.py
        "$task_python" -B run.py --smoke ;;
    full)
        cd "$task_home/ni26_s318_coral_fbl_20261117"
        "$task_python" -B replay/check.py > "$SLURM_SUBMIT_DIR/replay.log"
        grep -Fx OK "$SLURM_SUBMIT_DIR/replay.log"
        cd "$SLURM_SUBMIT_DIR"
        "$task_python" -B run.py ;;
    *) exit 2 ;;
esac
sha256sum -c source.sha256
