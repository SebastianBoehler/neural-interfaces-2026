#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-write449
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
sha256sum -c --status source.sha256
../venv/bin/python source.py smoke
../venv/bin/python source.py full
