#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:18:00
#SBATCH --job-name=ni26-aux431
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
orders=(within0 within1 within2 within3 full0 full1 runs0 runs1)
if [[ ${SLURM_ARRAY_TASK_ID:?} == 0 ]]; then
    "$python" replay/check.py > runs/aux431/replay.txt
fi
srun --cpu-bind=cores "$python" -B studies/18nov/aux431/run.py "${orders[$SLURM_ARRAY_TASK_ID]}"
