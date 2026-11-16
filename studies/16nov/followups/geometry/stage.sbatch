#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s295-stage
#SBATCH --output=logs/stage-%j.out
#SBATCH --error=logs/stage-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
export S295_PRODUCER_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s295_geometry_prod02_20261116
export S295_PRODUCER_SHA=aa7fe7746a31ba6387d88b4d6ef142acc6739d6cff81c165a1f8fcda27eaacfd
unset PYTHONPATH
sha256sum -c s295-source.sha256 >/dev/null
sha256sum -c s295-stage.sha256
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/geometry/stage.py
