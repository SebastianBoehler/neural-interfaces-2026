#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --job-name=ni26-hard-pool
#SBATCH --output=hard-pool-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
prior=$base/ni26_cross411b_20261118
mode=${1:?}
case "$mode" in
  old) release=$base/ni26_friday_f32x_external_cpu_v2_20261119 ;;
  new) release=$base/ni26_friday_c25_external_v1_20261119 ;;
  score) release=$base/ni26_friday_c25_external_v1_20261119 ;;
  *) exit 2 ;;
esac
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c --status source.sha256
(cd "$release" && sha256sum -c --status source.sha256 && sha256sum -c --status scoring.sha256)
grep -Fx OK "$prior/replay-proof.txt"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$release:$base/ni26_friday_c21e_native_v2_20261119:$base/ni26_friday_c20e_native_v4_20261119:$prior:$repo/experiments:$repo/2026-competition"
exec "$repo/.venv/bin/python" -B pooled.py "$mode"
