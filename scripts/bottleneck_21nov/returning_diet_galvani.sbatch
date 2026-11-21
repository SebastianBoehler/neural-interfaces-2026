#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
fixtures="$work/ni26_bottleneck_20261121"
qual="$work/ni26_last_push_20261121/src"
timing="$run/src/submissions/final_stack"
sif="$work/neural-interfaces-2026/runs/cloud/ni26_worker.sif"
ref=final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing_recognised_em_half
ref_sha=833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d
cap=v66_cp1_cap250=9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed
arm=${2:?}
name="${ref}_diet_${arm}"
sha=${3:?}
parent="$run/zips/$ref.zip"
candidate="$run/zips/$name.zip"
data="$qual/data/proteus_windows.npz"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NI26_QUAL_GPU='2080 Ti' NI26_MAX_MB=450
export BENCHOPT_DATA_HOME="$qual/data" PYTHONPATH="$qual/2026-competition"
export CS_OFFICIAL="$qual/2026-competition/tracks/bci_decoding"
container() { apptainer exec --nv --bind "$work" "$sif" python -B "$@"; }
case "${1:?}" in
    zero)
        container "$run/src/scripts/bottleneck_21nov/returning_diet_zero.py" \
            "$parent" "$candidate" "$fixtures" "$data" "$run/out/zero-$SLURM_JOB_ID.json" "$arm" "$sha"
        ;;
    pair)
        cd "$qual"
        container -c 'import sys
sys.path.insert(0, "friday")
import qualify_package as q
q.REF_SHA = "833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d"
q.REF_SCORE = 1.0
q.main()
' pair "$candidate" "$run/out/pair-$SLURM_JOB_ID" --ref-zip "$parent" --expect-sha "$sha"
        ;;
    timing)
        cd "$qual"
        container "$timing/e2e_timing.py" "$run/zips" "$data" "$run/out/e2e-$SLURM_JOB_ID.json" \
            "$cap" "$ref=$ref_sha" "$name=$sha" --unprofiled
        ;;
    *) exit 2 ;;
esac
