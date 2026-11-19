#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-c13-score
#SBATCH --output=c13-score-%j.out
set -euo pipefail
BASE=/mnt/lustre/work/gehler/gfh527
REPO=$BASE/neural-interfaces-2026
PRIOR=$BASE/ni26_cross411b_20261118
RELEASE=${SLURM_SUBMIT_DIR:?}
cd "$RELEASE"
sha256sum --quiet -c source.sha256
(cd score && sha256sum --quiet -c score.sha256)
(cd "$BASE" && sha256sum --quiet -c "$RELEASE/score/inputs.sha256")
grep -Fx OK "$PRIOR/replay-proof.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PRIOR:$REPO/experiments:$REPO/2026-competition"
exec "$REPO/.venv/bin/python" -B score/score.py "$RELEASE" "$@"
