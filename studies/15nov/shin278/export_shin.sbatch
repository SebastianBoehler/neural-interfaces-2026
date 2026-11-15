#!/bin/bash
#SBATCH --job-name=shin278
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/logs/shin278-%j.out
cd /mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
mkdir -p logs
.venv/bin/python -I studies/15nov/shin278/export_shin.py "$@"
