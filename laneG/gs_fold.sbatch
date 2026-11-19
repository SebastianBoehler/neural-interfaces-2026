#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=ni26-gs-fold
#SBATCH --output=logs/gs-fold-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527; repo=$base/neural-interfaces-2026; work=${SLURM_SUBMIT_DIR:?}; cd "$work"; mkdir -p pkg_gs
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --env "PYTHONDONTWRITEBYTECODE=1,NI26_ROOT=$repo,PYTHONPATH=/work" \
  --bind "$base:$base:ro,$work:/work:rw" "$repo/runs/cloud/ni26_worker.sif" python -B /work/gs_fold.py /work/pkg_gs
