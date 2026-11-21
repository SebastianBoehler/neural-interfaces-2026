#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:45:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
repo=$work/neural-interfaces-2026
parent=$work/ni26_ba_terminal_20261121/zips
lead=final_stack_hard_safe_st_team_b_nine25_tbgraph_fc
candidate=final_stack_hard_safe_st_team_b_ba_six_fc
niv=ba_six_fc_niv
lead_sha=7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07
cap_sha=9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NI26_QUAL_GPU='2080 Ti' NI26_MAX_MB=450
export BENCHOPT_DATA_HOME="$run/src/data" PYTHONPATH="$run/src/2026-competition"
container() { apptainer exec --nv --bind "$work" "$repo/runs/cloud/ni26_worker.sif" python -B "$@"; }
cd "$run/src"
stage=${1:?}
if [[ $stage == build ]]; then
    script=submissions/final_stack/patch_ba_six_fc.py
    container "$script" "$run/zips/$lead.zip" "$parent/final_stack_hard_safe_st_team_b_nine25_ba_terminal.zip" \
        "$run/zips/$candidate.zip" production | tee "$run/production.sha"
    container "$script" "$run/zips/$lead.zip" "$parent/ba_terminal_niv.zip" \
        "$run/zips/$niv.zip" niv | tee "$run/niv.sha"
    exit
fi
read -r sha _ < "$run/production.sha"
[[ ${#sha} == 64 ]]
case "$stage" in
    smoke|zero|identity)
        container submissions/final_stack/check_ba_six_fc.py "$stage" "$run/zips/$candidate.zip" "$sha" \
            data/proteus_windows.npz "$run/out/$stage-$SLURM_JOB_ID.json"
        ;;
    timing)
        container submissions/final_stack/e2e_timing.py "$run/zips" data/proteus_windows.npz \
            "$run/out/e2e-$SLURM_JOB_ID.json" "v66_cp1_cap250=$cap_sha" "$lead=$lead_sha" "$candidate=$sha"
        ;;
    pair)
        container friday/qualify_package.py pair "$run/zips/$candidate.zip" "$run/out/pair-$SLURM_JOB_ID" \
            --ref-zip "$run/zips/v66_cp1_cap250.zip" --expect-sha "$sha"
        ;;
    niv-*)
        read -r sha _ < "$run/niv.sha"
        [[ ${#sha} == 64 ]]
        exec bash "$run/src/scripts/breakthrough21/niv.sbatch" "${stage#niv-}" "$niv" "$sha" 71840
        ;;
    *) exit 2 ;;
esac
