#!/bin/bash
#SBATCH -A gehler
#SBATCH -p h100-ferranti,h100-preemptable-ferranti
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=6G
#SBATCH -t 4
#SBATCH -J ni26-all459
set -eu
sha256sum -c --status source.sha256
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case "$1" in
  smoke) ../venv/bin/python fit.py kernel 454 all smoke ;;
  train) ../venv/bin/python fit.py kernel "$((454+1000*SLURM_ARRAY_TASK_ID))" all ;;
  *) exit 2 ;;
esac
