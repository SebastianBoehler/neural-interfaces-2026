#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:06:00
#SBATCH --job-name=ni26-meta395
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export NI26_DEVICE=cuda OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
sha256sum -c source.sha256
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
if [[ "$1" == smoke ]]; then
 "$py" -B bodies.py 0 0 smoke
else
 i=${SLURM_ARRAY_TASK_ID:?}
 "$py" -B bodies.py "$((i/2))" "$((i%2))"
fi
