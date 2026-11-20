#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=ni26-Y-oracle
#SBATCH --output=y-%x-%j.out
# usage (cd to ni26_laneY_20261120): sbatch oracle.sbatch oracle_rigid|oracle_resrot|labelfree_resrot
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
py=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
$py -B laneY/$1.py out_$1.json
