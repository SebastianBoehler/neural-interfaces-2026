#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s312-cpu
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
sha256sum -c source.sha256 >/dev/null
export NI26_WS="$PWD" NI26_ROOT="$PWD" NI26_DEVICE=cpu NI26_OUT="$PWD/output"
export NI26_CACHE=/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_data1/cache
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
if [[ ${1:?} == probe ]]; then
 /usr/bin/time -f 'COMPLETE_CPU_PRODUCER_SECONDS=%e MAXRSS_KB=%M' srun --cpu-bind=cores .venv/bin/python run.py 80
else
 srun --cpu-bind=cores .venv/bin/python run.py "${SLURM_ARRAY_TASK_ID:?}"
fi
