#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-refit375
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
root=/mnt/lustre/work/gehler/gfh527
py=$root/neural-interfaces-2026/.venv/bin/python
sha256sum -c source.sha256
case ${1:?} in
    smoke) "$py" -B trajectory.py smoke 0 ;;
    shard) "$py" -B trajectory.py shard "${SLURM_ARRAY_TASK_ID:?}" ;;
    collect)
        (cd "$root/ni26_s318_coral_fbl_20261117" && "$py" replay/check.py) > replay.log
        grep -Fx OK replay.log
        "$py" -B collect.py ;;
    *) exit 2 ;;
esac
sha256sum -c source.sha256
