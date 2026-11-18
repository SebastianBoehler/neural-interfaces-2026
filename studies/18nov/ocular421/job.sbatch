#!/bin/bash
#SBATCH -A gehler
#SBATCH -p a100-galvani,2080-galvani
#SBATCH -c 1
#SBATCH --mem=3G
#SBATCH -t 6
#SBATCH -J ni26-eye421
set -eu
py=$PWD/../neural-interfaces-2026/.venv/bin/python
sha256sum -c --status sources.sha256
(cd ../ni26_s318_coral_fbl_20261117 && "$py" replay/check.py) > "replay-${SLURM_JOB_ID}.txt"
grep -x OK "replay-${SLURM_JOB_ID}.txt"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$py" analysis.py "$@"
