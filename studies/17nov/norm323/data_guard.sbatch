#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani
#SBATCH --gres=none
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=ni26-323-data-guard
#SBATCH --output=data-guard-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s323_dispatch_20261117
sha256sum -c data-guard.sha256 >/dev/null
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I data_guard.py
