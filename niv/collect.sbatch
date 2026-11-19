#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ni26-niv-collect
#SBATCH --output=logs/niv-collect-%A.out
# Usage: sbatch [--dependency=afterany:IDS] collect.sbatch A_TAG B_TAG [NULL_TAG]  -> out/report_A_vs_B.json (on cluster)
set -euo pipefail
a=${1:?}; b=${2:?}; null=${3:-}
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}
cd "$work"
extra=(); [ -n "$null" ] && extra=(--null "$null")
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --env PYTHONDONTWRITEBYTECODE=1 \
  --bind "$work:/work:rw,$repo/data/scherer2015/windows.npz:/external.npz:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B niv/collect.py out out/report_${a}_vs_${b}.json \
    --a "$a" --b "$b" "${extra[@]}" --cohort-a cohort_a.npz --data /external.npz
