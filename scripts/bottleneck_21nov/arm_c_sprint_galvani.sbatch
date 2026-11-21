#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani,a100-preemptable-galvani
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:09:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
work=/mnt/lustre/work/gehler/gfh527
repo="$work/neural-interfaces-2026"
qual="$work/ni26_last_push_20261121/src"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$qual/2026-competition"
export CS_OFFICIAL="$qual/2026-competition/tracks/bci_decoding"
export NI26_REPLAY_DATA="$repo/runs/replay_data"
python="$repo/.venv/bin/python"
task=${SLURM_ARRAY_TASK_ID:-48}
if [[ ${1:?} == smoke ]]; then
    "$python" -B "$repo/replay/check.py" | tee "v3-$SLURM_JOB_ID.log"
    grep -qx OK "v3-$SLURM_JOB_ID.log"
fi
(( task / 12 >= 4 && task / 12 <= 7 && task % 12 <= 9 ))
exec "$python" -B "$PWD/proteus-src/scripts/bottleneck_21nov/arm_c_sprint_proteus.py" held \
    "$qual/data/proteus_windows.npz" "$work/ni26_bottleneck_20261121" "$PWD" "$PWD/proteus-out" --task "$task"
