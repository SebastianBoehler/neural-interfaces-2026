#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani,a100-preemptable-galvani,v100-galvani
#SBATCH --exclude=galvani-cn109,galvani-cn111,galvani-cn121,galvani-cn122
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=ni26-s312-repr
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export NI26_WS="$PWD" NI26_ROOT="$PWD" NI26_DEVICE=cuda NI26_OUT="$PWD/output"
export NI26_CACHE=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_data1/cache
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
if [[ ${1:?} == guard_cpu ]]; then
 export NI26_DEVICE=cpu
 srun --cpu-bind=cores .venv/bin/python guard.py
elif [[ ${1:?} == guard ]]; then
 srun --cpu-bind=cores .venv/bin/python guard.py
else
 test -f guard_cuda.json
 srun --cpu-bind=cores .venv/bin/python run.py "${SLURM_ARRAY_TASK_ID:?}"
fi
