#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-preemptable-galvani,2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:12:00
#SBATCH --output=repeat-%j.log

set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
fixtures="$work/ni26_bottleneck_20261121"
qual="$work/ni26_last_push_20261121/src"
image="$work/neural-interfaces-2026/runs/cloud/ni26_worker.sif"
parent=final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing_recognised_em_half
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BENCHOPT_DATA_HOME="$qual/data" PYTHONPATH="$qual/2026-competition"
export CS_OFFICIAL="$qual/2026-competition/tracks/bci_decoding"
exec apptainer exec --nv --bind "$work" "$image" python -B "$run/src/scripts/bottleneck_21nov/arm_c_repeat_zero.py" \
    "$run/zips/$parent.zip" "$fixtures" "$qual/data/proteus_windows.npz" "$run/out/repeat-zero-$SLURM_JOB_ID.json"
