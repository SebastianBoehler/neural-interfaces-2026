#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s312-cpuwave
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export NI26_WS="$PWD"
.venv/bin/python -B dispatch_cpu.py "${1:?}"
