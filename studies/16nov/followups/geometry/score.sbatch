#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-s295-score
#SBATCH --output=logs/score-%j.out
#SBATCH --error=logs/score-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_WS="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
unset PYTHONPATH
sha256sum -c s295-source.sha256 >/dev/null
sha256sum -c s295-stage.sha256 >/dev/null
sha256sum -c s295-chain.sha256 >/dev/null
sha256sum -c s295-score.sha256
export S295_MANIFEST="$PWD/runs/s295_geometry/manifest.json"
S295_MANIFEST_SHA=$(sha256sum "$S295_MANIFEST" | cut -d' ' -f1)
export S295_MANIFEST_SHA
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -I studies/16nov/followups/geometry/score.py
