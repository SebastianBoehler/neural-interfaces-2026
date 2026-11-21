#!/usr/bin/env bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:50:00
# usage (sbatch, from the run dir): r1a_bases.sbatch MODE PERSON FIRST LAST [--smoke|--heldnew]
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/leonardo_work/AIFAC_P02_1636/ni26
source "$work/.venv312/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NI26_MAX_MB=100
export PYTHONPATH="$run/src/2026-competition"
cd "$run/src"
out=$run/out; [[ ${5:-} == --smoke ]] && out=$run/smoke
python -B scripts/overnight21/r1a_bases.py "$1" "$2" "$3" "$4" "$out" --zips "$work/cap250/runs/final-r1a-21nov/zips" \
  --data "$work/cap250/runs/final-r1a-21nov/src/data/proteus_windows.npz" ${5:+"$5"}
