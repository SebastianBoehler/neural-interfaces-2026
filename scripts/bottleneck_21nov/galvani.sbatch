#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:30:00
set -euo pipefail
run=${SLURM_SUBMIT_DIR:?}
work=/mnt/lustre/work/gehler/gfh527
repo="$work/neural-interfaces-2026"
src="$work/ni26_last_push_20261121/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$src/2026-competition"
export CS_OFFICIAL="$src/2026-competition/tracks/bci_decoding"
export NI26_REPLAY_DATA="$repo/runs/replay_data"
python="$repo/.venv/bin/python"
folds=(new_P01 new_P02 new_P03 new_P04 new_P05 new_P06 new_P07 new_P08 new_P09 new_P10 ret_P04 ret_P10)
rosters=(nine25 ba_six)
worker="$run/src/scripts/bottleneck_21nov/capture.py"
base="$src/data/proteus_windows.npz"
case "${1:?}" in
    smoke)
        "$python" -B "$repo/replay/check.py" | tee "$run/v3-$SLURM_JOB_ID.log"
        grep -qx OK "$run/v3-$SLURM_JOB_ID.log"
        "$python" -B "$worker" "$base" "$run" "$run/native-smoke" "${2:?}" "${3:?}" 71800 --smoke
        ;;
    capture)
        i=${SLURM_ARRAY_TASK_ID:?}
        roster=${rosters[i / 12 % 2]}
        fold=${folds[i % 12]}
        chunks=(71800 71802 71808 71810)
        first=${chunks[i / 24]}
        "$python" -B "$worker" "$base" "$run" "$run/native-out" "$roster" "$fold" "$first" "$((first + 1))"
        ;;
    *) exit 2 ;;
esac
