#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:18:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
qual="$work/ni26_last_push_20261121/src"
base="$work/ni26_bottleneck_20261121"
hss="$work/ni26_ba_six_fc_20261121/zips/final_stack_hard_safe_st.zip"
sif="$work/neural-interfaces-2026/runs/cloud/ni26_worker.sif"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$qual/2026-competition:$qual/2026-competition/tracks/bci_decoding"
export CS_OFFICIAL="$qual/2026-competition/tracks/bci_decoding"
options=(--direct)
if [[ ${1:?} == smoke ]]; then options+=(--smoke); fi
apptainer exec --nv --bind "$work" "$sif" python -B "$run/direct-src/scripts/fable_r8/audit.py" \
    "$qual/data/proteus_windows.npz" "$base" "$hss" "$run/out" "${SLURM_ARRAY_TASK_ID:?}" "${options[@]}"
