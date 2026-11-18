#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:18:00
#SBATCH --job-name=ni26-aux430
set -euo pipefail
source_root=/mnt/lustre/work/gehler/gfh527
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check source.sha256
cd "$source_root/ni26_s318_coral_fbl_20261117"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NI26_WS="$PWD"
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
srun --cpu-bind=cores "$source_root/neural-interfaces-2026/.venv/bin/python" -I -B "$work/run.py" "${orders[${SLURM_ARRAY_TASK_ID:?}]}"
