#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:06:00
#SBATCH --job-name=ni26-pooling367
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
base=/mnt/lustre/work/gehler/gfh527
py="$base/neural-interfaces-2026/.venv/bin/python"
sha256sum -c source.sha256
(
  cd "$base/ni26_selector363_v1_20261117"
  printf '%s  source.sha256\n' f18fc86a9a33ef4bf657d4e10f86759fa49987728ffe99cbe74e6b1c1acac41b | sha256sum -c -
  sha256sum -c source.sha256
)
"$py" -B pool.py --smoke
(
  cd "$base/ni26_s318_coral_fbl_20261117"
  "$py" replay/check.py
) > replay.log
grep -Fx OK replay.log
"$py" -B pool.py
sha256sum -c source.sha256
