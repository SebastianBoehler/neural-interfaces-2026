#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani,v100-galvani
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 6
#SBATCH -J ni26-collect451
set -euo pipefail
base=/mnt/lustre/work/gehler/gfh527
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum --quiet -c collector.sha256
sha256sum --quiet -c sources.sha256
(cd "$base" && sha256sum --quiet -c "$OLDPWD/anchors.sha256")
(cd "$base/ni26_s318_coral_fbl_20261117" && "$base/neural-interfaces-2026/.venv/bin/python" replay/check.py) > replay-proof.txt
grep -x OK replay-proof.txt
python3 collect.py
