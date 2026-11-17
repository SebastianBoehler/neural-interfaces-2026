#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s312-wave2short
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c schedule-source.sha256
ROOT=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2
(cd "$ROOT" && sha256sum -c source.sha256 >/dev/null)
"$ROOT/.venv/bin/python" -B wave2.py
