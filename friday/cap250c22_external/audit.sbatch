#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-c22e-audit
#SBATCH --output=c22e-audit-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
release=$base/ni26_friday_c22e_external_v1_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
sha256sum -c --status source.sha256
cd "$release"
sha256sum -c --status source.sha256
sha256sum -c --status scoring.sha256
grep -Fx OK "$prior/replay-proof.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$release:$base/ni26_friday_c21e_native_v2_20261119:$base/ni26_friday_c20e_native_v4_20261119:$prior:$repo/experiments:$repo/2026-competition"
exec "$repo/.venv/bin/python" -B "$work/selection_audit.py" "$work"
