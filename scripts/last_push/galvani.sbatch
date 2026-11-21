#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:40:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
repo="$work/neural-interfaces-2026"
zips="$work/ni26_ba_six_fc_20261121/zips"
roster=${2:?}
case "$roster" in
    nine25) parent=final_stack_hard_safe_st_team_b_nine25_tbgraph_fc; score=0.9959229391 ;;
    ba_six) parent=final_stack_hard_safe_st_team_b_ba_six_fc; score=1.0 ;;
    *) exit 2 ;;
esac
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${APPTAINERENV_CUDA_VISIBLE_DEVICES%%,*}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NI26_QUAL_GPU='2080 Ti' NI26_MAX_MB=450
export BENCHOPT_DATA_HOME="$run/src/data" PYTHONPATH="$run/src/2026-competition"
export CS_OFFICIAL="$run/src/2026-competition/tracks/bci_decoding"
export NI26_REPLAY_DATA="$repo/runs/replay_data"
container() { apptainer exec --nv --bind "$work" "$repo/runs/cloud/ni26_worker.sif" python -B "$@"; }
cd "$run/src"
script=scripts/last_push
export PYTHONPATH="$PWD/$script:$PYTHONPATH"
case "$1" in
    smoke)
        container "$repo/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
        grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
        container "$script/replay.py" production-smoke data/proteus_windows.npz "$roster" --zip "$zips/$parent.zip"
        ;;
    pair)
        arm=${3:?}
        candidate="$run/zips/${parent}_${arm}.zip"
        container "$script/patch.py" "$roster" "$arm" "$zips/$parent.zip" "$candidate"
        sha=$(sha256sum "$candidate" | cut -d' ' -f1)
        sha_parent=$(sha256sum "$zips/$parent.zip" | cut -d' ' -f1)
        container -c 'import sys
sys.path.insert(0, "friday")
import qualify_package as q
q.REF_SHA, q.REF_SCORE = sys.argv.pop(1), float(sys.argv.pop(1))
q.main()
' "$sha_parent" "$score" pair "$candidate" "$run/out/pair-$roster-$arm-$SLURM_JOB_ID" \
        --ref-zip "$zips/$parent.zip" --expect-sha "$sha"
        case "$arm" in
            rb_alpha|rb_mass|em_delay)
                ln -sfn "$zips/$parent.zip" "$run/zips/$parent.zip"
                container submissions/final_stack/e2e_timing.py "$run/zips" data/proteus_windows.npz \
                    "$run/out/runtime-$roster-$arm-$SLURM_JOB_ID.json" \
                    "$parent=$sha_parent" "${parent}_${arm}=$sha" --windows cpu --unprofiled
                ;;
        esac
        ;;
    *) exit 2 ;;
esac
