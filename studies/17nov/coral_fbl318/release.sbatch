#!/bin/bash
#SBATCH --account=gehler
#SBATCH --partition=cpu-galvani
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=ni26-318-release
#SBATCH --output=logs/318-release-%j.out
set -euo pipefail
cd /mnt/lustre/work/gehler/gfh527/ni26_s318_coral_fbl_20261117
sha256sum -c source318.sha256 >/dev/null
check=${1:?}
[[ "$(sacct -X -n -P -j "$check" -o State,ExitCode | head -1)" == 'COMPLETED|0:0' ]]
grep -qx OK "logs/318-check-$check.out"
grep -q 'OK: §318 zero/add/swap/FBL gate' "logs/318-check-$check.out"
active=$(squeue -h -u "$USER" -r -o '%i' | wc -l); running=$(squeue -h -u "$USER" -r -t RUNNING -o '%i' | wc -l)
if (( active + 45 > 272 || running + 4 > 32 )); then
 retry=$(sbatch --parsable --begin=now+10minutes studies/17nov/coral_fbl318/release.sbatch "$check")
 printf 'capacity %s/%s retry %s\n' "$active" "$running" "$retry" >> runs/coral304_inputs/s318-retries.log
 exit 0
fi
set -o noclobber
printf 'release %s\n' "$SLURM_JOB_ID" > runs/coral304_inputs/s318.intent
set +o noclobber
wave=$(sbatch --parsable --dependency=afterok:"$SLURM_JOB_ID" --array=1-23,25-31,33-39,41-47%4 studies/17nov/coral_fbl318/job.sbatch)
printf 'wave %s\n' "$wave" >> runs/coral304_inputs/s318.intent
score=$(sbatch --parsable --dependency=afterok:"$wave" studies/17nov/coral_fbl318/score.sbatch)
printf 'score %s\n' "$score" >> runs/coral304_inputs/s318.intent
