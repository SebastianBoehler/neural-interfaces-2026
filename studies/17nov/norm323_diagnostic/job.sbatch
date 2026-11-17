#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s323-diagnostic
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export DIAGNOSTIC_ROOT="$PWD" OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B -c "import audit, inputs, metrics; assert len(inputs.ARMS)==4"
srun --cpu-bind=cores "$PY" -B audit.py
