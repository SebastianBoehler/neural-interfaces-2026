#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=ni26-laneI
#SBATCH --output=logs/laneI-%A_%a.out
# usage (release dir with experiments/ laneI/ data/ runs/ 2026-competition/): sbatch --array=0-2 shrink.sbatch [FOLD ...]; task = seed
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LANEI_WORK=$PWD/out OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore
exec /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -u laneI/shrink_lopo.py "${SLURM_ARRAY_TASK_ID:?}" "$@"
