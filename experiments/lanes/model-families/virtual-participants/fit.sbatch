#!/bin/bash
# Lane V on Ferranti H100. usage (from the release root): sbatch laneV/fit.sbatch smoke | array 0-14 (task = arm_index*3 + seed)
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 90
#SBATCH -J ni26-laneV
#SBATCH -o logs/v-%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANEH_WORK=$PWD/work OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
PY=/home/gehler/gfh527/ni26/friday_s1c_training_20261118/runtime/bin/python
if [ "$1" = smoke ]; then  # code path only: 1 epoch, one fold, every arm
  mkdir -p work_smoke; export LANEH_WORK=$PWD/work_smoke LANEH_QUICK=1
  for a in V0 V1 V2 V2s V3; do "$PY" -u laneV/lopo_v.py $a 0 P02; done; exit 0
fi
arms=(V0 V1 V2 V2s V3); id=${SLURM_ARRAY_TASK_ID:?}
"$PY" -u laneV/lopo_v.py "${arms[$((id/3))]}" $((id%3))
