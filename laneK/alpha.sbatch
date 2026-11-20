#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-laneK4
#SBATCH --output=laneK4-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
b=/mnt/lustre/work/gehler/gfh527
exec $b/neural-interfaces-2026/.venv/bin/python -B alpha_peak.py $b/ni26_laneB5_20261120/data/proteus_windows.npz $b/ni26_laneB5_20261120/runs/shuffle_perms.npz alpha_peak.json
