#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s295-assemble
#SBATCH --output=logs/assemble-%j.out
#SBATCH --error=logs/assemble-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" NI26_ROOT="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
unset PYTHONPATH
printf '%s  %s\n' aa7fe7746a31ba6387d88b4d6ef142acc6739d6cff81c165a1f8fcda27eaacfd source.sha256 | sha256sum -c -
sha256sum -c source.sha256 >/dev/null
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/geometry/assemble.py
sha256sum -c source.sha256 >/dev/null
