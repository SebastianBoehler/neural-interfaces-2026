#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-c14-score
#SBATCH --output=c14-score-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
work=$PWD
sha256sum --check --status source.sha256
sha256sum --check --status scoring.sha256
(cd "$base" && sha256sum --check --status "$work/score_inputs.sha256")
grep -Fx OK "$prior/replay-proof.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$work/score_ref:$prior:$repo/experiments:$repo/2026-competition"
exec "$repo/.venv/bin/python" -B collect.py "$@"
