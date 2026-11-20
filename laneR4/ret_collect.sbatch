#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-R4-retcollect
#SBATCH --output=logs/R4retcollect-%j.out
# usage (release ret/ dir): RET_VAR=R1 sbatch --export=ALL [--dependency=afterany:ID] ret_collect.sbatch  -> summary_R1.json (BA table of every cell, gains, intervals)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
var=${RET_VAR:?}; export RET_ARMS=hardcb+$var
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B ret_collect.py out_$var summary_$var.json
