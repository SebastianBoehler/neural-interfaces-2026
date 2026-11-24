#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-laneI-ret-collect
#SBATCH --output=logs/laneI-ret-collect-%j.out
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527; work=${SLURM_SUBMIT_DIR:?}; cd "$work"
apptainer exec --cleanenv --containall --no-home --no-mount hostfs --bind "$base:$base:ro,$work:/work:rw" \
  "$base/neural-interfaces-2026/runs/cloud/ni26_worker.sif" python -B /work/collect_bn.py /work/out /work/summary.json
