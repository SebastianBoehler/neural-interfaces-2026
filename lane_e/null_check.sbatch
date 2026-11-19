#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-laneE-null
#SBATCH --output=logs/laneE-null-%A.out
# Usage (NI-V release dir): sbatch --dependency=afterany:IDS null_check.sbatch laneE_E0 cap250
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}
cd "$work"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --env PYTHONDONTWRITEBYTECODE=1 --bind "$work:/work:ro" \
  --pwd /work "$repo/runs/cloud/ni26_worker.sif" python -B laneE/null_check.py out "${1:?}" "${2:?}" | tee out/null_${1}_vs_${2}.txt
