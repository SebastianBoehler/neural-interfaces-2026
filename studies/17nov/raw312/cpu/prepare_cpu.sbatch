#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s312-cpuprep
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c setup.sha256
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B prepare_cpu.py
