#!/bin/bash
#SBATCH --job-name=ni26-aux317-fold
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --gres=none
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export NI26_ROOT="$PWD" NI26_DEVICE=cpu OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python
case ${1:?} in
 smoke) srun --cpu-bind=cores "$PY" fold.py 40 --smoke ;;
 work) srun --cpu-bind=cores "$PY" fold.py "${SLURM_ARRAY_TASK_ID:?}" ;;
 collect) export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; srun --cpu-bind=cores "$PY" collect.py ;;
 *) exit 2 ;;
esac
