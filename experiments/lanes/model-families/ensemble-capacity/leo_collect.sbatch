#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-laneE-leo-collect
#SBATCH --output=logs/leo-collect-%x-%A.out
# sbatch --account=ACC [--dependency=afterany:IDS] leo_collect.sbatch A_TAG B_TAG NULL_TAG  -> out/report_A_vs_B.json (cluster-side)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; unset PYTHONPATH; module load python/3.11.7 >/dev/null 2>&1 || true
PY=${LEO_REPO:-/leonardo_work/AIFAC_P02_1636/ni26}/.venv/bin/python
a=${1:?}; b=${2:?}; null=${3:-}; extra=(); [ -n "$null" ] && extra=(--null "$null")
$PY -I -B niv_leo/collect.py out out/report_${a}_vs_${b}.json --a "$a" --b "$b" "${extra[@]}" --cohort-a data/cohort_a.npz --data data/windows.npz
