#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s315-stage
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c representation-source.sha256 >/dev/null
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B studies/17nov/representation_chain/stage.py "${1:?}"
