#!/bin/bash
#SBATCH -A gehler
#SBATCH -p cpu-galvani
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J ni26-R4-zero
#SBATCH -o logs/zero-%j.out
# usage (release ret/ dir): sbatch check_zero.sbatch ; zips/ built by ret_build.sbatch. R0z is the code-path control and the reference of the bitwise comparison
set -euo pipefail
R=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
cd "${SLURM_SUBMIT_DIR:?}"; mkdir -p logs
apptainer exec --bind /mnt/lustre/work/gehler/gfh527 --env BENCHOPT_DATA_HOME=$R/data $R/runs/cloud/ni26_worker.sif \
  bash -c "cd $R && python -B $SLURM_SUBMIT_DIR/../laneR4/check_zero.py /mnt/lustre/work/gehler/gfh527/ni26_laneT_20261120/time/final_stack_hardcb.zip $(for a in R0z R1 R2 R3 R4 R2s R2q; do printf '%s ' $SLURM_SUBMIT_DIR/zips/final_stack_hardcb_$a.zip; done)"
