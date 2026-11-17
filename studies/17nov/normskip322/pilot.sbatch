#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-322-pilot
#SBATCH --output=pilot-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s322_normskip_20261117
sha256sum -c source322.sha256 >/dev/null
export PYTHONPATH="$PWD:$PWD/experiments" NI26_DEVICE=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B pilot.py
