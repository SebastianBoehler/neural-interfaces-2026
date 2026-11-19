#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:04:00
#SBATCH --job-name=ni26-c21g-external
#SBATCH --output=c21g-external-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
owner=$base/ni26_friday_c20e_native_v4_20261119
native=$base/ni26_friday_c21e_native_v2_20261119
shared=$base/ni26_friday_c21s_native_v1_20261119
work=${SLURM_SUBMIT_DIR:?}
cd "$work"
(cd "$shared" && sha256sum --check --status source.sha256 && sha256sum --check --status scoring.sha256)
[[ $(readlink -f source.sha256) == "$shared/source.sha256" ]]
sha256sum --check --status scoring.sha256
(cd "$native" && sha256sum --check --status source.sha256)
(cd "$owner" && sha256sum --check --status scoring.sha256)
(cd "$base" && sha256sum --check --status ""$owner"/parents.sha256" && sha256sum --check --status ""$owner"/score_inputs.sha256")
grep -Fx OK "$prior/replay-proof.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$shared:$native:$owner:$prior:$repo/experiments:$repo/2026-competition"
exec "$repo/.venv/bin/python" -B score_gate.py
