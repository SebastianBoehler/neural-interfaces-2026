#!/bin/bash
#SBATCH --account=AIFAC_P02_1636
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
archive=${1:?}
sha=${2:?}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$run/src/2026-competition" NI26_QUAL_GPU=A100
source "$WORK/ni26/.venv312/bin/activate"
cd "$run/src"
python -B friday/qualify_package.py pair "$run/zips/$archive" \
    "$run/out/pair-${archive%.zip}-$SLURM_JOB_ID" --ref-zip "$run/zips/v66_cp1_cap250.zip" --expect-sha "$sha"
