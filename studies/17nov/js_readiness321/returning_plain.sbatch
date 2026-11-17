#!/bin/bash
#SBATCH --job-name=ni26-js321-returning-plain-pilot
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:06:00
#SBATCH --gres=gpu:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
families=(shallow_ea shallow_raw fbcnet_ea_tf fbcnet_raw_tf)
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python returning_plain.py "${families[${SLURM_ARRAY_TASK_ID:?}]}" P04_s23 0
