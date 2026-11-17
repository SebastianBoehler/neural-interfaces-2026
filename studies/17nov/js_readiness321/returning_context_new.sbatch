#!/bin/bash
#SBATCH --job-name=ni26-js321-new-source-context
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:04:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
folds=(P04_s23 P10_s23)
out="$PWD/context_${SLURM_ARRAY_TASK_ID:?}"
mkdir "$out"
cp source.sha256 s281_fold_stage_diag.py "$out/"
cd "$out"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python ../returning_context.py "${folds[$SLURM_ARRAY_TASK_ID]}"
