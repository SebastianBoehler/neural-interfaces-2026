#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani,2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:16:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
work=/mnt/lustre/work/gehler/gfh527
setup=$work/ni26_ba_six_fc_20261121
repo=$work/neural-interfaces-2026
pinned=$setup/niv-results/ba_six_fc_niv_fp16_turing
mode=${1:?}
task=${SLURM_ARRAY_TASK_ID:-0}
sha=
bind="$work,$setup/src/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$setup/niv-rows.tsv.gz:/rows.tsv.gz:ro"
if [[ $mode == smoke-new ]]; then task=20; fi
if [[ $mode == smoke-ret ]]; then task=5; fi
if [[ $mode == ret || $mode == smoke-ret ]]; then
    name="hss_n25_k$task.zip"
    sha=$(awk -v name="$name" '$2 == name {print $1}' "$setup/enrolled.sha256")
    [[ ${#sha} == 64 ]]
    bind+=",$work/ni26_niv_finalA_20261121/zips/enrolled/$name:/enrolled.zip:ro"
fi
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
gpu=(--nv)
if [[ $mode == build || $mode == report-* ]]; then gpu=(); fi
apptainer exec "${gpu[@]}" --bind "$bind" "$repo/runs/cloud/ni26_worker.sif" \
    python -B scripts/bottleneck_21nov/returning_em_niv.py "$mode" "$task" "$sha" "$pinned"
