#!/bin/bash
#SBATCH --job-name=ni26-js321-returning-other-pilot
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
families=(fb fbl fbraw)
family=${families[${SLURM_ARRAY_TASK_ID:?}]}
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python returning_other.py "$family" P04_s23 0
