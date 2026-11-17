#!/bin/bash
#SBATCH --job-name=ni26-aux317-release
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:55:00
#SBATCH --gres=none
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export NI26_ROOT="$PWD" PYTHONDONTWRITEBYTECODE=1
.venv/bin/python dispatch.py "${1:?}"
