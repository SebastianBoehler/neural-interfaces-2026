#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=2080-galvani,2080-preemptable-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=ni26-Y-err
#SBATCH --output=y-%x-%j.out
# usage (cd to ni26_laneY_20261120): sbatch err_member.sbatch out.npz [smoke]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B laneY/err_member.py "$@"
