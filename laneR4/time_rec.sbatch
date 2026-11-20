#!/bin/bash
#SBATCH -A gehler
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -J ni26-R4-time
#SBATCH -o logs/time-%x-%j.out
# usage (release ret/ dir): [R4_TIME_ARMS='R0z R5 R1a R0z'] sbatch --export=ALL -p PART --gres=gpu:X:1 time_rec.sbatch ; R0z (= final stack) first and last as the paired reference, RECOGNISED real sessions
set -euo pipefail
R=/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026
cd "${SLURM_SUBMIT_DIR:?}"; mkdir -p logs
arms=${R4_TIME_ARMS:-"R0z R1 R2 R3 R4 R2s R2q R0z"}
for a in $arms; do z=zips/final_stack_hardcb_$a.zip
  apptainer exec --nv --bind /mnt/lustre/work/gehler/gfh527 --env BENCHOPT_DATA_HOME=$R/data $R/runs/cloud/ni26_worker.sif \
    bash -c "cd $R && python $SLURM_SUBMIT_DIR/../laneR4/time_rec.py $SLURM_SUBMIT_DIR/$z cuda" 2>&1 | grep '^{' | tee -a logs/timerec-$SLURM_JOB_ID.log || true
done
