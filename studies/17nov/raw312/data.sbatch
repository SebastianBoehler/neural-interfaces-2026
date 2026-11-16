#!/bin/bash
#SBATCH --job-name=ni26-s312-data
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --gres=none
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256
export NI26_ROOT=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
export NI26_CACHE="$PWD/cache" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [[ ${1:?} == parity ]]; then
 srun --cpu-bind=cores "$NI26_ROOT/.venv/bin/python" data.py parity
else
 printf -v person 'P%02d' "$((SLURM_ARRAY_TASK_ID+1))"
 srun --cpu-bind=cores "$NI26_ROOT/.venv/bin/python" data.py export --person "$person"
fi
