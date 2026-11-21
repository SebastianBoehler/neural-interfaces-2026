#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:14:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
fixtures="$work/ni26_bottleneck_20261121"
qual="$work/ni26_last_push_20261121/src"
sif="$work/neural-interfaces-2026/runs/cloud/ni26_worker.sif"
ref=final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing_recognised_em_half_diet_c
ref_sha=cba9b208fbff7ce416efd8b5ce68ae1ad0771980d84e754433bd33e9996d0bca
arm=${2:?}
name="${ref}_${arm}"
sha=${3:?}
parent="$run/zips/$ref.zip"
candidate="$run/zips/$name.zip"
data="$qual/data/proteus_windows.npz"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NI26_QUAL_GPU='2080 Ti' NI26_MAX_MB=450
export BENCHOPT_DATA_HOME="$qual/data"
export PYTHONPATH="$qual/2026-competition:$run/src/scripts/last_push:$run/src/scripts/bottleneck_21nov"
export CS_OFFICIAL="$qual/2026-competition/tracks/bci_decoding"
container() { apptainer exec --nv --bind "$work" "$sif" python -B "$@"; }
case "${1:?}" in
    zero)
        container "$run/scripts/bottleneck_21nov/arm_c_sprint_zero.py" \
            "$parent" "$candidate" "$fixtures" "$data" "$run/out/zero-$SLURM_JOB_ID.json" "$arm" "$sha"
        ;;
    pair)
        cd "$qual"
        container -c 'import sys
sys.path.insert(0, "friday")
import qualify_package as q
q.REF_SHA = sys.argv.pop(1)
q.REF_SCORE = 1.0
q.main()
' "$ref_sha" pair "$candidate" "$run/out/pair-$SLURM_JOB_ID" --ref-zip "$parent" --expect-sha "$sha"
        ;;
    timing)
        cd "$qual"
        container "$run/src/submissions/final_stack/e2e_timing.py" "$run/zips" "$data" "$run/out/e2e-$SLURM_JOB_ID.json" \
            "$ref=$ref_sha" "$name=$sha" --unprofiled
        ;;
    *) exit 2 ;;
esac
