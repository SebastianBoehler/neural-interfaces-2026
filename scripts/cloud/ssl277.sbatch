#!/bin/bash
# s277 jobs on Leonardo, from $WORK/ni26_ssl. Usage: sbatch scripts/cloud/ssl277.sbatch prep | fold   (fold: use --array=0-9; smoke: sbatch --export=ALL,SMOKE=smoke ... with --array=0)
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
module load python/3.11.7 >/dev/null 2>&1; unset PYTHONPATH; . .venv/bin/activate
export OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore SSL_WS=$PWD
date -u; md5sum studies/14nov/ssl/*.py experiments/protocol.py
case $1 in
  prep) python -I studies/14nov/ssl/prep_ext.py ;;
  fold2) python -I studies/14nov/ssl/ssl_form2.py ${SLURM_ARRAY_TASK_ID:-0} ${SMOKE:-} ;;
  fold) python -I studies/14nov/ssl/ssl_fold.py ${SLURM_ARRAY_TASK_ID:-0} ${SMOKE:-} ;;
esac; rc=$?; echo EXIT $rc; date -u; exit $rc
