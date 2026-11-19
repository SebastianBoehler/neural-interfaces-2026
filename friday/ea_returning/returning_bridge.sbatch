#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:14:00
#SBATCH --job-name=ni26-ear-zero
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
owner=$base/ni26_aux431_20261118
labels=$base/ni26_friday_ea_matrix_20261118
refits=$base/ni26_friday_causal_rebuild_20261118/dependents_full0_v2
sha256sum -c --status bridge_source.sha256
sha256sum -c --status bridge_inputs.sha256
grep -Fx OK "$owner/runs/aux431/replay.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$labels:$refits:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
"$repo/.venv/bin/python" -B returning_bridge.py "$1"
