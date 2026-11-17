#!/bin/bash
#SBATCH --job-name=ni26-js321-active
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" build_draft.py
"$PY" package_guard.py
(cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 && sha256sum -c source318.sha256 >/dev/null)
for order in within0 within1 within2 within3 full0 full1 runs0 runs1; do
  NI26_ROOT=/mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117 "$PY" study_recipe.py "$order"
done
