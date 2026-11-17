#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-318-wave1
#SBATCH --output=logs/318-wave1-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117
sha256sum -c source318.sha256 >/dev/null
sha256sum -c source318-waves.sha256 >/dev/null
[[ "$(sacct -X -n -P -j 3116565 -o State,ExitCode | head -1)" == 'COMPLETED|0:0' ]]
grep -qx OK logs/318-check-3116565.out
grep -q 'OK: §318 zero/add/swap/FBL gate' logs/318-check-3116565.out
active=$(squeue -h -u "$USER" -r -o '%i' | wc -l); running=$(squeue -h -u "$USER" -r -t RUNNING -o '%i' | wc -l)
if (( active + 23 > 272 || running + 4 > 32 )); then
 retry=$(sbatch --parsable --begin=now+5minutes studies/17nov/coral_fbl318/wave1.sbatch)
 printf 'wave1 capacity %s/%s retry %s\n' "$active" "$running" "$retry" >> runs/coral304_inputs/s318-waves-retries.log
 exit 0
fi
set -o noclobber
printf 'wave1 dispatcher %s\n' "$SLURM_JOB_ID" > runs/coral304_inputs/s318-waves.intent
set +o noclobber
wave=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=25-31,33-39,41-47%4 studies/17nov/coral_fbl318/job.sbatch)
printf 'wave1 %s\n' "$wave" >> runs/coral304_inputs/s318-waves.intent
callback=$(sbatch --parsable --dependency=afterok:"$wave" studies/17nov/coral_fbl318/wave2.sbatch)
printf 'wave2 dispatcher %s\n' "$callback" >> runs/coral304_inputs/s318-waves.intent
