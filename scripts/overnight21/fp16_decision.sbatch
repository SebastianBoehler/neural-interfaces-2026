#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:30:00
set -euo pipefail

run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
parent=$work/ni26_ba_six_fc_20261121
sif=$work/neural-interfaces-2026/runs/cloud/ni26_worker.sif
trial=ba_six_fc_niv_fp16_turing
trial_sha=0804a2f272d4fb4b6215cfcd7f282f0c813cbe4b4888f3f6b5f38f74964d6160
niv() {
    bash "$run/src/scripts/breakthrough21/niv.sbatch" "$1" "$trial" "$trial_sha" "$2"
}

export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
container() { apptainer exec --nv --bind "$work" "$sif" python -B "$@"; }

case "${1:?}" in
    smoke-new|smoke-ret)
        niv "$1" 71840
        out="niv-results/$trial"
        container "$run/fp16_niv_smoke.py" "$1" "$parent/$out" "$run/$out"
        niv "$1" 71848
        ;;
    new|ret|report)
        niv "$1" 71848
        ;;
    timing|timing-smoke)
        src=$parent/unprofiled-source/submissions/final_stack
        export BENCHOPT_DATA_HOME="$run/src/data"
        export PYTHONPATH="$src:$run/src/submissions/final_stack:$run/src/2026-competition"
        export NI26_QUAL_GPU='2080 Ti' NI26_MAX_MB=450
        cap=v66_cp1_cap250=9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed
        candidate=final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing=890c2beec60344280f21b7206e99378dc71047a90b2e5db0876986b636b40cad
        cd "$run/src"
        args=("$run/zips" data/proteus_windows.npz "$run/out/$1-$SLURM_JOB_ID.json" "$cap" "$candidate" --unprofiled)
        if [[ $1 == timing-smoke ]]; then
            container -c 'import e2e_timing as timing
import sys
plans = timing.C.plans
timing.C.plans = lambda *args: dict(list(plans(*args).items())[:3])
sys.argv.pop(0)
timing.main()
' e2e_timing.py "${args[@]}"
        else
            container "$src/e2e_timing.py" "${args[@]}"
        fi
        ;;
    *) exit 2 ;;
esac
