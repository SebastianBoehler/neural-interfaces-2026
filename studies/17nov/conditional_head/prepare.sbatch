#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=ni26-s326-prepare
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s326_conditional_head_20261117
source studies/17nov/conditional_head/env.sh
"$PY" -B -I studies/17nov/conditional_head/prepare.py
