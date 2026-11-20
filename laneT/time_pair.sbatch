#!/bin/bash
#SBATCH -A gehler
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J b1-timepair
#SBATCH -o time-%x-%j.out
# usage: sbatch -p PART --gres=gpu:X:1 time_pair.sbatch DIR CAND_ZIP   cap250.zip (A) and CAND_ZIP (B) timed A B B A on one node, unrecognised sessions
set -euo pipefail
R=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
cd "$1"; cand=$2
for z in cap250.zip "$cand" "$cand" cap250.zip; do
  apptainer exec --nv --bind /mnt/lustre/work/gehler/gfh527 --env BENCHOPT_DATA_HOME=$R/data $R/runs/cloud/ni26_worker.sif \
    bash -c "cd $R && python $R/../ni26_laneT_20261120/laneT/time_ft.py $PWD/$z cuda 4" 2>&1 | grep '^{' | tee -a pair-$SLURM_JOB_ID.log || true
done
