#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --job-name=fable-shift
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
source_root=${1:?}; shift
root=/mnt/lustre/work/gehler/gfh527
repo=$root/neural-interfaces-2026
lead=$root/ni26_final_confirm_20261121/zips/final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2
export NI26_TORCH_THREADS=2 NI26_ROOT="$source_root"
export CS_OFFICIAL=$repo/2026-competition/tracks/bci_decoding
export PYTHONPATH="$CS_OFFICIAL:$repo/2026-competition"
nvidia-smi -L
apptainer exec --nv --bind "$root" "$repo/runs/cloud/ni26_worker.sif" \
    python -B "$source_root/scripts/fable_shiftenrol/worker.py" "$@" \
    --lead "$lead" --inputs "$run/inputs" --data "$repo/data/proteus_windows.npz"
