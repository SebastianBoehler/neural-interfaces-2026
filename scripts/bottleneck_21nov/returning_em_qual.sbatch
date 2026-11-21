#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
repo="$work/neural-interfaces-2026"
src="$work/ni26_last_push_20261121/src"
parent="$work/ni26_ba_six_fc_20261121/zips/final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing.zip"
candidate=${2:?}
sha=${3:?}
arm=${4:?}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NI26_QUAL_GPU='2080 Ti' NI26_MAX_MB=450
export BENCHOPT_DATA_HOME="$src/data" PYTHONPATH="$src/2026-competition"
export CS_OFFICIAL="$src/2026-competition/tracks/bci_decoding"
container() { apptainer exec --nv --bind "$work" "$repo/runs/cloud/ni26_worker.sif" python -B "$@"; }
case "${1:?}" in
    zero)
        container "$run/src/scripts/bottleneck_21nov/returning_em_zero.py" \
            "$parent" "$candidate" "$run" "$src/data/proteus_windows.npz" "$run/out/returning-em-zero-$SLURM_JOB_ID.json" "$arm" "$sha"
        ;;
    pair)
        cd "$src"
        container -c 'import sys
sys.path.insert(0, "friday")
import qualify_package as q
q.REF_SHA = "890c2beec60344280f21b7206e99378dc71047a90b2e5db0876986b636b40cad"
q.REF_SCORE = 1.0
q.main()
' pair "$candidate" "$run/out/returning-em-pair-$SLURM_JOB_ID" --ref-zip "$parent" --expect-sha "$sha"
        ;;
    *) exit 2 ;;
esac
