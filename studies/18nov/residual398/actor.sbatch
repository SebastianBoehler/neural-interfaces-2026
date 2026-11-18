#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani
#SBATCH --nodelist=galvani-cn102
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:07:00
#SBATCH --job-name=ni26-resid398
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_DEVICE=cpu OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
case "$1" in
actor)
    people=(P01 P02 P03 P09)
    "$py" -B actor398.py "${people[${SLURM_ARRAY_TASK_ID:?}]}" ;;
smoke)
    (cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) >replay.log
    grep -Fx OK replay.log
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$py" -B full398.py smoke 0 ;;
full)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    "$py" -B full398.py full "$SLURM_ARRAY_TASK_ID" ;;
score) "$py" -B score398.py ;;
*) exit 2 ;;
esac
