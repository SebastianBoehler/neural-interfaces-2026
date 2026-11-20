#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-laneN-sum
#SBATCH --output=logs/laneN-sum-%A.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}
cd "$work"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --env PYTHONDONTWRITEBYTECODE=1 --bind "$work:/work:rw" --pwd /work \
  "$repo/runs/cloud/ni26_worker.sif" python -B laneN_summarize.py out out/laneN_summary.json out/laneN_summary.md \
  cap250 hard hardcbB hardcbB_tttlite5 hardcbB_tttsparse20 hardcbB_tttlight hardcbB_tttheavy
