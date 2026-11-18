#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani,2080-preemptable-galvani
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -t 18
#SBATCH -J ni26-stages456
set -euo pipefail
root=/mnt/lustre/work/gehler/gfh527
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum -c --status source.sha256
cd "$root/ni26_s318_coral_fbl_20261117"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NI26_WS="$PWD"
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
srun --cpu-bind=cores "$root/neural-interfaces-2026/.venv/bin/python" -I -B "$work/stages.py" "${orders[${SLURM_ARRAY_TASK_ID:?}]}"
