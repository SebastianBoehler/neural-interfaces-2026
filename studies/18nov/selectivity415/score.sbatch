#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani
#SBATCH -c 1
#SBATCH --mem=3G
#SBATCH -t 4
#SBATCH -J ni26-shin415s
set -eu
py=$PWD/../neural-interfaces-2026/.venv/bin/python
(cd ../ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) > replay.txt
grep -x OK replay.txt
OPENBLAS_NUM_THREADS=1 "$py" score.py
