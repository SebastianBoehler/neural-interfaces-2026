#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:06:00
#SBATCH --job-name=ni26-ear-warm
#SBATCH --output=warm-%A_%a.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
repo=$base/neural-interfaces-2026
archive=$base/ni26_aux431_20261118
source=$base/ni26_js321_20261117_returning_banks_complete1
sha256sum -c --status fresh_warm_source.sha256
sha256sum -c --status peer_inputs.sha256
(cd "$source" && sha256sum -c --status source.sha256)
grep -Fx OK "$archive/runs/aux431/replay.txt"
names=(fbraw sinc)
family=${names[${SLURM_ARRAY_TASK_ID:?}]}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$source:$repo/2026-competition:$repo/2026-competition/tracks/bci_decoding"
mkdir "$family"
cd "$family"
"$repo/.venv/bin/python" -B ../fresh_peer_refit.py "$family"
