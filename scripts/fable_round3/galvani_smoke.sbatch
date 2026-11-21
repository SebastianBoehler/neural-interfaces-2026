#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --job-name=fable3-public-smoke
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
root=/mnt/lustre/work/gehler/gfh527
repo=$root/neural-interfaces-2026
lead=$root/ni26_final_confirm_20261121/zips/final_stack_hard_safe_st_team_b_nine25_tbgraph.zip
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export CS_OFFICIAL=$repo/2026-competition/tracks/bci_decoding
export PYTHONPATH="$CS_OFFICIAL:$repo/2026-competition"
nvidia-smi -L
apptainer exec --nv --bind "$run,$root" "$repo/runs/cloud/ni26_worker.sif" \
    python -B "$run/src/scripts/fable_round3/worker.py" public-smoke --lead "$lead" --data "$repo/data/proteus_windows.npz"
