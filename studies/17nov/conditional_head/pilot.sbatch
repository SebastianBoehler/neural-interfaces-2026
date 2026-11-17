#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-s326-pilot
#SBATCH --output=logs/pilot-%j.out
#SBATCH --error=logs/pilot-%j.err
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s326_conditional_head_20261117
source studies/17nov/conditional_head/env.sh
"$PY" -B -I studies/17nov/conditional_head/pilot.py
"$PY" -B -I studies/17nov/conditional_head/score.py
