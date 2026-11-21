#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:16:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
work=/mnt/lustre/work/gehler/gfh527
setup="$work/ni26_ba_six_fc_20261121"
repo="$work/neural-interfaces-2026"
diet="$work/ni26_diet_decomp_20261121"
pinned="$work/ni26_em_half_final_20261121/niv-results"
baseline="$work/ni26_quarter_final_20261121/niv-results"
mode=${1:?}
task=${SLURM_ARRAY_TASK_ID:-5}
out="out/$mode"
mkdir -p "$out"
name="hss_n25_k$task.zip"
sha=$(awk -v name="$name" '$2 == name {print $1}' "$setup/enrolled.sha256")
[[ ${#sha} == 64 ]]
bind="$work,$setup/src/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$setup/niv-rows.tsv.gz:/rows.tsv.gz:ro"
bind+=",$work/ni26_niv_finalA_20261121/zips/enrolled/$name:/enrolled.zip:ro"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTHONPATH="$diet/src/scripts/last_push"
exec apptainer exec --nv --bind "$bind" "$repo/runs/cloud/ni26_worker.sif" \
    python -B scripts/bottleneck_21nov/diet_decomposition_niv_tape.py \
    "$mode" "$task" "$sha" "$baseline" "$pinned" "$out" --paired-a100 \
    --sprint-pin "$diet/niv-tape-score-a100/c"
