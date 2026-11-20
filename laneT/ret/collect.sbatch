#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-laneT-retcollect
#SBATCH --output=logs/retcollect-%j.out
# usage: RET_VAR=lite5 sbatch --export=ALL collect.sbatch   (RET_VAR unset: out, summary.json, arms hardcb+ftlight+ftheavy)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
var=${RET_VAR:-}; sfx=${var:+_$var}
if [ -n "$var" ]; then export RET_ARMS=hardcb+ft$var; fi
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B ttt_collect.py out$sfx summary$sfx.json
