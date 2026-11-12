#!/bin/bash
#SBATCH --job-name=ni26-replay
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/diag/leo_replay-%j.out
# Official ingestion + scoring replay of one package on a Leonardo A100 (COMPET_DEVICE=cuda), as the Galvani replay.sbatch.
# Not the worker image. Usage: sbatch scripts/cloud/leo_replay.sbatch runs/sealed/PKG
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate
export BENCHOPT_DATA_HOME="$PWD/data" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
nvidia-smi --query-gpu=name,driver_version --format=csv; lscpu | grep -E "Model name|^CPU\(s\)"
PKG="${1%/}"; python scripts/replay_package.py "$PKG" "${PKG}_replay_leo_$SLURM_JOB_ID" cuda
