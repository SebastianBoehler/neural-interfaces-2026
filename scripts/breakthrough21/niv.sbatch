#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
repo=$work/neural-interfaces-2026
enrolled=$work/ni26_niv_finalA_20261121/zips/enrolled
stage=${1:?}
tag=${2:?}
sha=${3:?}
first=${4:?}
task=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$stage" == smoke-new ]]; then task=20; fi   # Scherer G on order 0
if [[ "$stage" == smoke-ret ]]; then task=5; fi
bind="$work,$run/src/2026-competition:/official:ro,$repo/data/scherer2015/windows.npz:/external.npz:ro,$run/niv-rows.tsv.gz:/rows.tsv.gz:ro"
extra=()
if [[ "$stage" == ret || "$stage" == smoke-ret ]]; then
    name="hss_n25_k$task.zip"
    receipt=$(awk -v name="$name" '$2 == name {print $1}' "$run/enrolled.sha256")
    [[ ${#receipt} == 64 ]]
    extra=(--enrolled-sha "$receipt")
    bind+=",$enrolled/$name:/enrolled.zip:ro"
fi
cd "$run"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
apptainer exec --nv --bind "$bind" "$repo/runs/cloud/ni26_worker.sif" python -B niv_pair.py \
    "$stage" "$run/niv-results/$tag" "$first" --task "$task" --sha "$sha" \
    --base "$run/zips/final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip" \
    --trial "$run/zips/$tag.zip" "${extra[@]}"
