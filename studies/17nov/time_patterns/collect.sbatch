#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s314-collect
#SBATCH --output=logs/s314-collect-%j.out
#SBATCH --error=logs/s314-collect-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c s314-source.sha256 >/dev/null
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/17nov/time_patterns/assemble.py
"$PY" -B -I studies/17nov/time_patterns/score.py
