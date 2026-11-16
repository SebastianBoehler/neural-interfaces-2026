#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s315-link312
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s315_link312_20261117
sha256sum -c source.sha256
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B link312.py
