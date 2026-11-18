#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-pair388src
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
py=../venv/bin/python
"$py" -B source388.py "$1"
