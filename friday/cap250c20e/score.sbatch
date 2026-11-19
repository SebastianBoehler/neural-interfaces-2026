#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-c20e-score
#SBATCH --output=c20e-score-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum --check --status source.sha256
sha256sum --check --status scoring.sha256
(cd "$base" && sha256sum --check --status "$work/parents.sha256" && sha256sum --check --status "$work/score_inputs.sha256")
grep -Fx OK "$prior/replay-proof.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$prior:$repo/experiments:$repo/2026-competition"
exec "$repo/.venv/bin/python" -B score.py
