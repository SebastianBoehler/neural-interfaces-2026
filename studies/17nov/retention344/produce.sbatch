#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --gres=none
#SBATCH --array=0-3%4
#SBATCH --job-name=ni26-s344-retention
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export RETENTION_ROOT="$PWD" NI26_DEVICE=cpu CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
arms=(A B C D)
export ARM="${arms[$SLURM_ARRAY_TASK_ID]}"
export SMOKE_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_retention344_smoke_v3_20261117
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B run344.py
