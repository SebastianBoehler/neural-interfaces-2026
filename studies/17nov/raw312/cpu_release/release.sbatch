#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s312-cpugate
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_release_20261117
sha256sum -c release-source.sha256
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B release.py
