#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --gres=none
#SBATCH --job-name=ni26-s315-prepare
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s315_setup_20261117
sha256sum -c setup.sha256 >/dev/null
PY=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python
"$PY" -B representation_chain/prepare.py "${1:?}" "${2:?}"
