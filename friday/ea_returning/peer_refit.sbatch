#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-ear-peer
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
owner=$base/ni26_aux431_20261118
sha256sum -c --status peer_source.sha256
sha256sum -c --status peer_inputs.sha256
grep -Fx OK "$owner/runs/aux431/replay.txt"
case "${1:-$SLURM_ARRAY_TASK_ID}" in
    0|fb) name=fb; stem=fb ;;
    1|fbraw) name=fbraw; stem=raw ;;
    *) exit 2 ;;
esac
device=${2:-cpu}
grep -E "^$stem s0 P10_s23 n_own=192 n_te=385 fit [0-9]+s" "$repo/runs/diag/pdeep/logs/pd-3069177_0.out" | cut -d'|' -f1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
mkdir "$name-$device"
cd "$name-$device"
"$repo/.venv/bin/python" -B ../peer_refit.py "$name" "$device"
