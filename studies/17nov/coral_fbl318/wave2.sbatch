#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-318-wave2
#SBATCH --output=logs/318-wave2-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117
sha256sum -c source318.sha256 >/dev/null
sha256sum -c source318-waves.sha256 >/dev/null
active=$(squeue -h -u "$USER" -r -o '%i' | wc -l); running=$(squeue -h -u "$USER" -r -t RUNNING -o '%i' | wc -l)
if (( active + 25 > 272 || running + 4 > 32 )); then
 retry=$(sbatch --parsable --begin=now+5minutes studies/17nov/coral_fbl318/wave2.sbatch)
 printf 'wave2 capacity %s/%s retry %s\n' "$active" "$running" "$retry" >> runs/coral304_inputs/s318-waves-retries.log
 exit 0
fi
set -o noclobber
printf 'wave2 dispatcher %s\n' "$SLURM_JOB_ID" > runs/coral304_inputs/s318-wave2.intent
set +o noclobber
wave=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=1-23%4 studies/17nov/coral_fbl318/job.sbatch)
printf 'wave2 %s\n' "$wave" >> runs/coral304_inputs/s318-wave2.intent
score=$(sbatch --parsable --dependency=afterok:"$wave" studies/17nov/coral_fbl318/score.sbatch)
printf 'score %s\n' "$score" >> runs/coral304_inputs/s318-wave2.intent
