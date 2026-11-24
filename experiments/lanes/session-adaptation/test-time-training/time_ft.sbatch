#!/bin/bash
#SBATCH -A gehler
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -J ni26-laneT-time
#SBATCH -o time-%j.out
# usage: sbatch -p PART --gres=gpu:X:1 time_ft.sbatch DIR ZIP1 ZIP2 ...   (zips inside DIR, run twice in the order given)
set -euo pipefail
R=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
cd "$1"; shift
for rep in 1 2; do for z in "$@"; do
  apptainer exec --nv --bind /mnt/lustre/work/gehler/gfh527 --env BENCHOPT_DATA_HOME=$R/data $R/runs/cloud/ni26_worker.sif \
    bash -c "cd $R && python $R/../ni26_laneT_20261120/laneT/time_ft.py $PWD/$z cuda 4" 2>&1 | tee -a full-$SLURM_JOB_ID.log | grep '^{' || true
done; done
