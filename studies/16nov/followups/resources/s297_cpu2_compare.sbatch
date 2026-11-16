#!/usr/bin/env bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=ni26-s297-filtercheck
#SBATCH --output=logs/filtercheck-%j.out
#SBATCH --error=logs/filtercheck-%j.err
set -eu
export NI26_WS="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
printf '%s  %s\n' 985ff8632f3b119b655336bc69bd02479b417223cc65ba2e7fb6f560636e40bf compare_resource.py | sha256sum -c -
/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python -B -I compare_resource.py
